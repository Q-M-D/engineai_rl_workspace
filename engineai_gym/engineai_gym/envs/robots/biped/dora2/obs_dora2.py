from engineai_gym.envs.base.obs.obs import Obs


class ObsDora2(Obs):
    def commands(self):
        return self.env.commands
