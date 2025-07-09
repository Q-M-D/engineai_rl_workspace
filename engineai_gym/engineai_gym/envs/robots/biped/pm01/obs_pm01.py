from engineai_gym.envs.base.obs.obs import Obs


class ObsPm01(Obs):
    def dof_vel(self, lag=False):
        if lag:
            try:
                return (
                    self.env.domain_rands.domain_rands_type_obs_lag.lagged_dof_vel
                    * self.env.dof_vel_ignored_joints
                )
            except:
                return None
        else:
            return self.env.dof_vel

    def com_displacements(self):
        return self.env.domain_rands.domain_rands_type_rigid_body.com_displacements
