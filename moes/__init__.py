from . import attributes
from . import ave_speed
from . import link_speed
from . import slow_movement
from . import speed_reduction


# These three keep one semicolon-separated number per 200-ft piece.
# Link length, travel time, and measured speed share cv_dashboard.link_speed.
POINT_MEASURES = (
    ave_speed,
    speed_reduction,
    slow_movement,
)

MEASURES = POINT_MEASURES


def empty_state():
    return {module.TABLE_NAME: None for module in MEASURES}


def add_sequence(state, sequence, segments):
    state[ave_speed.TABLE_NAME] = ave_speed.reduce(
        state[ave_speed.TABLE_NAME], ave_speed.partial(sequence)
    )
    state[speed_reduction.TABLE_NAME] = speed_reduction.reduce(
        state[speed_reduction.TABLE_NAME], speed_reduction.partial(sequence)
    )
    state[slow_movement.TABLE_NAME] = slow_movement.reduce(
        state[slow_movement.TABLE_NAME], slow_movement.partial(sequence)
    )
    return state


def publish(state, segments):
    for module in MEASURES:
        module.write(state[module.TABLE_NAME], segments)
