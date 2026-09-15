"""Label-preserving voxel edits driven by exact cubical vertex-link defects."""
from itertools import product
from numbers import Integral
import numpy as np

_CORNERS=np.array(list(product((0,1),repeat=3)))
_OFFSETS=np.vstack((np.eye(3,dtype=int),-np.eye(3,dtype=int)))


def _valid_link(mask):
    # The boundary link is a graph on the six signed lattice-axis rays.
    # A manifold surface vertex has one cycle, with degree two at every ray.
    if mask in (0,255):return True
    edges=[]
    for i,c in enumerate(_CORNERS):
        for axis in range(3):
            j=i^(1<<(2-axis))
            if j>i and ((mask>>i)&1)!=((mask>>j)&1):
                uv=[a for a in range(3) if a!=axis]
                edges.append(tuple(2*a+int(c[a]) for a in uv))
    adjacency={}
    for a,b in edges:
        adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
    if any(len(v)!=2 for v in adjacency.values()):return False
    seen=set();todo=[next(iter(adjacency))]
    while todo:
        v=todo.pop()
        if v not in seen:seen.add(v);todo.extend(adjacency[v])
    return len(seen)==len(adjacency)


_VALID=np.array([_valid_link(m) for m in range(256)])


def _defects(block):
    v=block.ravel()
    return {int(g) for g in np.unique(v) if g>=0 and not _VALID[int(np.sum((v==g)*(1<<np.arange(8))))]}


def detect_voxel_topology(labels):
    """Return defective (padded block origin -> grain IDs) cubical vertex links.

    Includes box faces/edges/corners, point pinches, alternating edge contacts,
    and disconnected complementary sectors. This tests each grain's closed
    cubical boundary, not whether a genuine multi-grain junction should exist.
    """
    a=np.asarray(labels)
    if a.ndim!=3 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0):
        raise ValueError('labels must be a nonempty nonnegative integer 3D array')
    if a.max()>np.iinfo(np.int64).max:raise ValueError('Grain IDs must fit int64')
    p=np.pad(a.astype(np.int64),1,constant_values=-1)
    blocks=np.lib.stride_tricks.sliding_window_view(p,(2,2,2)).reshape(-1,8)
    result={};shape=tuple(np.array(a.shape)+1)
    weights=1<<np.arange(8)
    for corner in range(8):
        g=blocks[:,corner]
        first=(g>=0)&(~np.any(blocks[:,:corner]==g[:,None],axis=1))
        ids=np.flatnonzero(first)
        masks=((blocks[ids]==g[ids,None])*weights).sum(axis=1)
        for i in ids[~_VALID[masks]]:
            origin=tuple(map(int,np.unravel_index(i,shape)))
            result.setdefault(origin,set()).add(int(g[i]))
    return result


def clean_voxel_topology(labels, enabled=True, max_passes=20, max_changes=5000,
                         max_volume_fraction=.2, protect_grains_up_to=0,
                         preserve_components=False, max_pair_trials=20000):
    """Greedily reduce vertex-link defects without deleting a grain ID.

    Only voxels adjacent to detected defects are considered. Every edit strictly
    reduces the global defect count (all affected links are recomputed). Volume
    budgets bound per-grain net count changes. Component preservation is optional
    because removing legitimate contacts may require merging/splitting components.
    Unresolved defects are reported explicitly; no success is inferred from stop.
    """
    a=np.asarray(labels).copy()
    for name,v in [('max_passes',max_passes),('max_changes',max_changes),('protect_grains_up_to',protect_grains_up_to),('max_pair_trials',max_pair_trials)]:
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) or v<0:raise ValueError(name+' must be a nonnegative integer')
    if not np.isfinite(max_volume_fraction) or not 0<=max_volume_fraction<=1:raise ValueError('Invalid volume fraction')
    if not isinstance(enabled,(bool,np.bool_)) or not isinstance(preserve_components,(bool,np.bool_)):raise ValueError('Switches must be boolean')
    candidates=detect_voxel_topology(a)
    initial_count=sum(map(len,candidates.values()))
    ids,counts=np.unique(a,return_counts=True);initial=dict(zip(map(int,ids),map(int,counts)));current=initial.copy()
    allowance={g:max(1,int(n*max_volume_fraction)) if max_volume_fraction else 0 for g,n in initial.items()}
    p=np.pad(a.astype(np.int64),1,constant_values=-1);changes=[];history=[]
    reason='disabled' if not enabled else 'maximum passes reached'
    if preserve_components:
        from scipy.ndimage import label
        components={g:label(a==g)[1] for g in initial}
    pair_trials=0
    def pair_repair(positions):
        nonlocal pair_trials
        for pos in positions:
            donor=int(p[pos])
            if initial[donor]<=protect_grains_up_to:continue
            origins1={tuple(np.array(pos)-c) for c in _CORNERS}
            neighbors=[tuple(np.array(pos)+d) for d in _OFFSETS]
            recipients=sorted({int(p[q]) for q in neighbors}-{donor,-1})
            for recipient in recipients:
                p[pos]=recipient
                for q in neighbors:
                    if not all(1<=q[j]<=a.shape[j] for j in range(3)):continue
                    donor2=int(p[q])
                    if initial[donor2]<=protect_grains_up_to:continue
                    targets=sorted(({int(p[tuple(np.array(q)+d)]) for d in _OFFSETS}|{donor})-{donor2,-1})
                    origins=origins1|{tuple(np.array(q)-c) for c in _CORNERS}
                    before=sum(len(candidates.get(o,())) for o in origins)
                    for recipient2 in targets:
                        if pair_trials>=max_pair_trials:p[pos]=donor;return None
                        pair_trials+=1
                        delta={g:0 for g in (donor,recipient,donor2,recipient2)}
                        delta[donor]-=1;delta[recipient]+=1;delta[donor2]-=1;delta[recipient2]+=1
                        if any(current[g]+d<1 or abs(current[g]+d-initial[g])>allowance[g] for g,d in delta.items()):continue
                        p[q]=recipient2
                        after={o:_defects(p[tuple(slice(x,x+2) for x in o)]) for o in origins}
                        gain=before-sum(map(len,after.values()))
                        valid=gain>0
                        if valid and preserve_components:
                            view=p[1:-1,1:-1,1:-1]
                            valid=all(label(view==g)[1]==components[g] for g in delta)
                        p[q]=donor2
                        if valid:
                            p[pos]=donor
                            return pos,donor,recipient,q,donor2,recipient2,after,delta,gain
                p[pos]=donor
        return None
    for iteration in range(max_passes if enabled else 0):
        accepted=0
        positions=sorted({tuple(np.array(o)+c) for o in candidates for c in _CORNERS
                          if np.all(np.array(o)+c>=1) and np.all(np.array(o)+c<=a.shape)})
        for pos in positions:
            if len(changes)>=max_changes:break
            donor=int(p[pos])
            if initial[donor]<=protect_grains_up_to or current[donor]<=1:continue
            if abs(current[donor]-1-initial[donor])>allowance[donor]:continue
            origins=[tuple(np.array(pos)-c) for c in _CORNERS]
            before=sum(len(candidates.get(o,())) for o in origins)
            if not before:continue
            recipients=sorted({int(p[tuple(np.array(pos)+d)]) for d in _OFFSETS}-{donor,-1})
            best=None
            for recipient in recipients:
                if abs(current[recipient]+1-initial[recipient])>allowance[recipient]:continue
                p[pos]=recipient
                after={o:_defects(p[tuple(slice(x,x+2) for x in o)]) for o in origins}
                gain=before-sum(map(len,after.values()))
                if gain>0 and (best is None or gain>best[0]):
                    valid=True
                    if preserve_components:
                        view=p[1:-1,1:-1,1:-1]
                        valid=all(label(view==g)[1]==components[g] for g in (donor,recipient))
                    if valid:best=(gain,recipient,after)
                p[pos]=donor
            if best is None:continue
            gain,recipient,after=best;p[pos]=recipient
            current[donor]-=1;current[recipient]+=1;accepted+=1
            for o,defects in after.items():
                if defects:candidates[o]=defects
                else:candidates.pop(o,None)
            changes.append(dict(voxel=[int(x-1) for x in pos],from_grain=donor,to_grain=recipient,removed_defects=gain))
        if not accepted and candidates and len(changes)+2<=max_changes and pair_trials<max_pair_trials:
            pair=pair_repair(positions)
            if pair is not None:
                pos,donor,recipient,q,donor2,recipient2,after,delta,gain=pair
                p[pos]=recipient;p[q]=recipient2
                for g,d in delta.items():current[g]+=d
                for o,defects in after.items():
                    if defects:candidates[o]=defects
                    else:candidates.pop(o,None)
                group=len(changes)
                for v,old,new in [(pos,donor,recipient),(q,donor2,recipient2)]:
                    changes.append(dict(voxel=[int(x-1) for x in v],from_grain=old,to_grain=new,pair_group=group,removed_defects=gain))
                accepted=2
        history.append(dict(pass_number=iteration+1,accepted=accepted,remaining=sum(map(len,candidates.values()))))
        if not candidates:reason='no defects remain';break
        if len(changes)>=max_changes:reason='maximum changes reached';break
        if not accepted:reason='no admissible bounded repair';break
    result=p[1:-1,1:-1,1:-1].astype(a.dtype)
    assert np.array_equal(np.unique(result),ids)
    final=detect_voxel_topology(result)
    assert final==candidates
    report=dict(enabled=bool(enabled),initial_defects=initial_count,remaining_defects=sum(map(len,final.values())),
                accepted_changes=len(changes),net_changed_voxels=int(np.count_nonzero(result!=a)),
                all_grain_ids_preserved=True,preserve_components=bool(preserve_components),stop_reason=reason,passes=history,pair_trials=pair_trials,
                unresolved=[dict(vertex=list(o),grains=sorted(gs)) for o,gs in sorted(final.items())],
                grain_voxel_changes={str(g):current[g]-initial[g] for g in initial if current[g]!=initial[g]})
    return result,report,changes
