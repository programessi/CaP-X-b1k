import ast

from capx.utils.launch_utils import _extract_code


def test_extract_code_preserves_unfenced_multiline_function_call():
    response = """RESULT = pick_and_place_visual_object(
    "x2_pick_place_blue_cube",
    [0.370, 0.055, 0.921],
    prompts=["blue cube", "blue block", "blue box"],
    table_name="x2_pick_place_table",
    sim_place_correction_steps=4,
)"""

    blocks = _extract_code(response)

    assert blocks == [response]
    ast.parse(blocks[0])
    assert blocks[0].rstrip().endswith(")")


def test_extract_code_trims_trailing_prose_after_multiline_call():
    response = """Here is the code:
RESULT = pick_and_place_visual_object(
    "x2_pick_place_blue_cube",
    [0.370, 0.055, 0.921],
)
This should complete the task."""

    blocks = _extract_code(response)

    assert len(blocks) == 1
    assert "This should complete" not in blocks[0]
    ast.parse(blocks[0])
    assert blocks[0].rstrip().endswith(")")
