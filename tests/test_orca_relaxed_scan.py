import enum

if not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = _StrEnum

from avogadro_generators.orca import generateInputFile
from avogadro_generators.orca.input_blocks import Basis, ElProp, SCF


def resolved_block_default(kwd):
    if kwd.options is not None and kwd.default is not None:
        return kwd.options[kwd.default]
    if kwd.default is None and kwd._dtype is str:
        return ""
    return kwd.default


def default_options() -> dict:
    options = {
        "Title": "",
        "Filename Base": "job",
        "Processor Cores": 1,
        "Memory": 4,
        "Calculation Type": "Opt",
        "Theory": "r2SCAN-3c",
        "Basis": "def2-TZVP",
        "Charge": 0,
        "Multiplicity": 1,
        "Solvent": "",
        "Solvation Model": "CPCM",
        "basic_disp_corr": "",
        "basic_print_mos": True,
        "basic_constrain": False,
        "basic_print_level": "NormalPrint",
        "basic_use_symmetry": False,
        "basic_simple_keywords": "",
        "excited_state_method": "None",
        "excited_num_states": 3,
        "excited_target_state": 1,
        "Basis_pople": "",
        "Basis_def2": "",
        "Basis_cc": "",
        "Basis_jensen": "",
        "Basis_relativistic": "",
    }

    for enum in (SCF, Basis, ElProp):
        for kwd in enum:
            options[kwd.get_json_key()] = resolved_block_default(kwd)

    options["Basis_AUXJ"] = ""
    options["Basis_AUXJK"] = ""
    options["Basis_AUXC"] = ""

    return options


def test_orca_emits_relaxed_scan_constraints():
    options = default_options()
    options["basic_constrain"] = True

    input_json = {
        "options": options,
        "cjson": {
            "atoms": {"elements": {"number": [8, 6, 6, 8]}},
            "constraints": [
                [1.35, 0, 1],
                {
                    "value": 180.0,
                    "atoms": [0, 1, 2, 3],
                    "scan": {"initial": 180.0, "end": 360.0, "steps": 19},
                },
            ],
        },
    }

    generated_input, warnings, syntax_groups = generateInputFile(input_json)

    assert warnings == []
    assert "%geom\n" in generated_input
    assert "    Constraints \n" in generated_input
    assert "        { B 0 1 1.350000 C } \n" in generated_input
    assert "    Scan\n" in generated_input
    assert "        D 0 1 2 3 = 180.000000, 360.000000, 19\n" in generated_input
    assert syntax_groups == ["default"]
