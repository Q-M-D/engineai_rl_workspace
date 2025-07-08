from .tester_type_base import TesterTypeBase


class TesterZeroCommands(TesterTypeBase):
    def set_goals(self) -> None:
        self.env.commands[:] = 0
        return self.env.goal_dict
