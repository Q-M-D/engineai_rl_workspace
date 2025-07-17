from __future__ import annotations

import torch
from engineai_rl.algos.ppo import Ppo


class PpoContact(Ppo):
    def __init__(self, networks, policy_cfg, env, device="cpu", **kwargs):
        self.contact_net = networks["contact"]
        super().__init__(networks, policy_cfg, env, device=device, **kwargs)

    def act(self, inputs):
        contact_feat = self.contact_net(inputs["contact"]).detach()
        actor_inputs = inputs.copy()
        actor_inputs["actor"] = torch.cat((inputs["actor"], contact_feat), dim=-1)
        self.transition.actions = self.actor_critic.act(actor_inputs).detach()
        self.transition.values = self.actor_critic.evaluate(actor_inputs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.inputs = actor_inputs
        return self.transition.actions

    def compute_returns(self, inputs):
        contact_feat = self.contact_net(inputs["contact"]).detach()
        actor_inputs = inputs.copy()
        actor_inputs["actor"] = torch.cat((inputs["actor"], contact_feat), dim=-1)
        last_values = self.actor_critic.evaluate(actor_inputs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
