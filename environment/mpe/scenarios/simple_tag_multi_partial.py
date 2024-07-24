import numpy as np
from environment.mpe.core import World, Agent, Landmark
from environment.mpe.scenario import BaseScenario


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


class Scenario(BaseScenario):
    tower_positions = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    def make_multi_world(self, num_agents, num_good_agents, obs_radius, init_radius,
                         shaped_reward, collide_reward, collide_reward_once, watch_tower):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.obs_radius = obs_radius
        world.init_radius = init_radius
        world.shaped_reward = shaped_reward
        world.collide_reward = collide_reward
        world.collide_reward_once = collide_reward_once
        world.watch_tower = watch_tower
        num_good_agents = num_good_agents
        num_adversaries = num_agents - num_good_agents
        num_landmarks = 2 + (4 if watch_tower else 0)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i < 2:
                landmark.name = 'landmark %d' % i
                landmark.collide = True
                landmark.movable = False
                landmark.size = 0.2
                landmark.boundary = False
            else:
                landmark.name = f'watch_tower {i - 2}'
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.05
                landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            agent.caught = False
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < 2:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0, 0, 1.0])
                landmark.alpha = 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.init_radius, +world.init_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                if i < 2:
                    landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                else:
                    landmark.state.p_pos = np.array(self.tower_positions[i - 2])
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        adversaries = self.adversaries(world)
        if world.shaped_reward:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))

        if agent.collide and world.collide_reward:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if world.shaped_reward:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for agt in agents:
                if not (world.collide_reward_once and agt.caught):
                    rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agt.state.p_pos))) for a in adversaries])
            # for adv in adversaries:
            #     rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide and world.collide_reward:
            for agt in agents:
                if any(self.is_collision(agt, adv) for adv in adversaries) and not (world.collide_reward_once and agt.caught):
                    rew += 10
                # for adv in adversaries:
                #     if self.is_collision(ag, adv):
                #         rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        if world.watch_tower and any(self.is_collision(agent, tower) for tower in world.landmarks[2:]):
            obs_radius = 1e9
        else:
            obs_radius = world.obs_radius
        entity_pos = []
        entity_vis = []
        for entity in world.landmarks:
            if not entity.boundary:
                d = distance(entity.state.p_pos, agent.state.p_pos)
                if d <= obs_radius:
                    entity_vis.append(np.ones(1))
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_vis.append(np.zeros(1))
                    entity_pos.append(np.zeros(world.dim_p))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        other_vis = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            d = distance(other.state.p_pos, agent.state.p_pos)
            if d <= obs_radius:
                other_vis.append(np.ones(1))
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_vel.append(other.state.p_vel)
            else:
                other_vis.append(np.zeros(1))
                other_pos.append(np.zeros(world.dim_p))
                other_vel.append(np.zeros(world.dim_p))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]
                              + entity_vis + entity_pos + other_vis + other_pos + other_vel)

    def info(self, agent, world):
        if world.watch_tower:
            return {'keypoint_visited': [any(self.is_collision(agent, tower) for tower in world.landmarks[2:])]}
        return {}

    def done(self, agent, world):
        if not world.collide_reward_once:
            return False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        return all(agt.caught or any(self.is_collision(agt, adv) for adv in adversaries) for agt in agents)

    def post_step(self, world):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for agt in agents:
            if (not agt.caught) and any(self.is_collision(agt, adv) for adv in adversaries):
                agt.caught = True

