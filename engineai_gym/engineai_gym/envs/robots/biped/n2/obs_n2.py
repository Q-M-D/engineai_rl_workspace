from engineai_gym.envs.base.obs.obs import Obs


class ObsN2(Obs):
    def commands(self):
        return self.env.commands
