from .base_api import list_apis, register_api

# ---------------------------------------------------------------------------
# Base Franka APIs (shared utilities)
# ---------------------------------------------------------------------------
from .franka.control import FrankaControlApi
from .franka.control_privileged import FrankaControlPrivilegedApi
from .franka.control_reduced import FrankaControlApiReduced
from .franka.control_reduced_skill_library import FrankaControlApiReducedSkillLibrary

register_api("FrankaControlPrivilegedApi", FrankaControlPrivilegedApi)
register_api("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3=True))
register_api("FrankaControlApiReduced", FrankaControlApiReduced)
register_api("FrankaControlApiReducedSkillLibrary", FrankaControlApiReducedSkillLibrary)

# Multi-turn variant
register_api(
    "FrankaControlMultiPrivilegedApi",
    lambda env: FrankaControlPrivilegedApi(env, multi_turn=True),
)

# ---------------------------------------------------------------------------
# Real Franka APIs
# ---------------------------------------------------------------------------
register_api("FrankaRealReducedSkillLibraryControlApi", lambda env: FrankaControlApiReducedSkillLibrary(env, tcp_offset=[0.0, 0.0, -0.157], real=True))
register_api("FrankaRealControlApi", lambda env: FrankaControlApi(env, tcp_offset=[0.0, 0.0, -0.157], real=True))

# ---------------------------------------------------------------------------
# R1Pro Behavior APIs
# ---------------------------------------------------------------------------
try:
    from .r1pro.control import R1ProControlApi
    register_api("R1ProControlApi", lambda env: R1ProControlApi(env, use_sam3=True))
    register_api("R1ProControlApiSAM2", lambda env: R1ProControlApi(env, use_sam3=False))
except ImportError:
    print("R1Pro not installed, skipping R1Pro APIs")
