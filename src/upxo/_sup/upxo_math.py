from collections import deque
import random


def MATH_ruc(low, high, gap):
    '''
    MATH_ruc: random_uniform_constrained
    Source: https://stackoverflow.com/questions/75813482/generate-random-numbers-in-a-range-while-keeping-a-minimum-distance-between-valu
    '''
    slots, freespace = deque([(low, high)]), high - low
    while slots:
        x = random.uniform(0, freespace)
        while True:
            slotlow, slothigh = slots[0]
            slotspace = slothigh - slotlow
            if x < slotspace:
                slots.popleft()
                freespace -= slotspace
                xlow, xhigh = x - gap, x + gap
                if xlow > 0:
                    slots.append((slotlow, slotlow + xlow))
                    freespace += xlow
                if xhigh < slotspace:
                    slots.appendleft((slotlow + xhigh, slothigh))
                    freespace += slotspace - xhigh
                yield x + slotlow
                break
            x -= slotspace
            slots.rotate(-1)
    return slots
