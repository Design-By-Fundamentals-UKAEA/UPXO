# validation/

Scripts that compare UPXO with mtex. They need mtex (pyMTEX) installed; nothing in the shipped
`upxo` package imports mtex or this folder. The tests that use these scripts' output are in
`tests/validation_mtex/`, which runs without mtex (see its README).

| Script | Needs mtex | Writes | Purpose |
|---|---|---|---|
| `generate_reference.py` | yes | `tests/validation_mtex/reference/ks_mtex.json` | Frozen mtex values the test suite checks against, with the mtex/numpy/Python versions and the date |
| `generate_report.py` | yes | `validation/reports/*.md`, `validation/reports/provenance.json` | The markdown tables and comparison numbers used to write the mtex-validation wiki pages |

Run both from the repository root:

```
python validation/generate_reference.py
python validation/generate_report.py
```

`reports/provenance.json` records the mtex, numpy and Python versions, the date and the UPXO
git commit the report was generated from. Re-running after a code or mtex change updates the
numbers; a wiki page that quotes one of these tables should be regenerated and re-checked
rather than hand-edited.

No real EBSD file, file path or measured orientation appears in these scripts or in anything
they write. Every input is a synthetic Euler angle or a construction from first principles
(the four {111} normals, the Kurdjumov-Sachs vector pairs).
