""" 
A basic agent (player) for the Starcraft II module.

Code based on: https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

:author Dr. Kevin Brewer
:version (Summer 2019) (still valid Winter 2021)
:python (3.6) (and 3.8)
"""
# ------ import section ------
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random as r
from newZergAgent import ZergAgent

# ------ CONSTANTS ------
FUNCTIONS = actions.FUNCTIONS
NUM_SIMS = 20
NUM_TESTS = 3
# ------ main function ------
def main(unused_argv):
    """
    The main function that trains the agent and runs the Starcraft simulations.


    Args:
        unused_argv: unused arguments
    Returns:
        None
    """
    agent = ZergAgent()  # our agent class defined above
    try:
        i = 0
        while (
            i < NUM_SIMS
        ):  # run one simulation (can change parameters to run multiple)
            i += 1
            print("+++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++")
            print("Running simulation:", i)
            print("+++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++")
            with sc2_env.SC2Env(
                map_name="MoveToBeacon",  # the map to be played on
                players=[
                    sc2_env.Agent(sc2_env.Race.terran),  # our agent
                ],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True,
                ),  # allows us to use the feature.units interface
                # (easier to use in our agent)
                step_mul=16,  # related to how many Actions Per game time Minute. this value gives about 150 APM
                # a lower value would give more APM. 8 is the default
                game_steps_per_episode=0,  # keep the game going until a resolution - i.e., no time limit
                visualize=True,
            ) as env:

                # the following code is 'boilerplate' to get the game going
                agent.setup(env.observation_spec(), env.action_spec())
                curr_obs = env.reset()
                obs = curr_obs[0]  # curr_obs[0] == obs
                agent.reset()
                while True:  # run the simulation timesteps until done (defined above)
                    # returned_state = agent.get_state(timesteps[0])
                    # agent.remember(timesteps[0], 1, 1, timesteps[0])
                    step_actions = [agent.step(obs)]  # curr_obs[0] == obs
                    actions = step_actions[0]
                    
                    print(
                        "Action function id:",
                        int(str(FUNCTIONS[actions[0]]).split("/")[0]),
                    )
                    
                    if obs.last():
                        print("+==================+")
                        print(
                            "...Final Score:",
                            obs.observation.score_cumulative[0],
                        )
                        print("+==================+")
                        break

                    next_obs = env.step(
                        step_actions
                    )  # note we need to use step_actions, not actions
                    obs_next = next_obs[0]
                    agent.remember(obs, actions, 0, obs_next)
                    if int(str(FUNCTIONS[actions[0]]).split("/")[0]) == 7:
                        agent.remember(obs, actions, 0, obs_next)
                        agent.remember(obs, actions, 0, obs_next)
                        agent.remember(obs, actions, 0, obs_next)
                        agent.remember(obs, actions, 0, obs_next)
                        agent.remember(obs, actions, 0, obs_next)
                        agent.remember(obs, actions, 0, obs_next)
                        agent.remember(obs, actions, 0, obs_next)
                    obs = obs_next
                    agent.learn()

    except KeyboardInterrupt:  # stop running with a keyboard interrupt
        pass

    agent.save()

    # Run the tests with the trained NN
    agent.run_tests()
    try:
        i = 0
        while (
            i < NUM_TESTS
        ):  # run one simulation (can change parameters to run multiple)
            i += 1
            print("+++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++")
            print("Running NN test simulation:", i)
            print("+++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++")
            with sc2_env.SC2Env(
                map_name="MoveToBeacon",  # the map to be played on
                players=[
                    sc2_env.Agent(sc2_env.Race.terran),  # our agent
                ],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True,
                ),  # allows us to use the feature.units interface
                # (easier to use in our agent)
                step_mul=16,  # related to how many Actions Per game time Minute. this value gives about 150 APM
                # a lower value would give more APM. 8 is the default
                game_steps_per_episode=0,  # keep the game going until a resolution - i.e., no time limit
                visualize=True,
            ) as env:

                # the following code is 'boilerplate' to get the game going
                agent.setup(env.observation_spec(), env.action_spec())
                curr_obs = env.reset()
                agent.reset()
                while True:  # run the simulation timesteps until done (defined above)
                    # returned_state = agent.get_state(timesteps[0])
                    # agent.remember(timesteps[0], 1, 1, timesteps[0])
                    step_actions = [agent.step(curr_obs[0])]  # curr_obs[0] == obs
                    if curr_obs[0].last():
                        print("+==================+")
                        print(
                            "...Final Score:",
                            curr_obs[0].observation.score_cumulative[0],
                        )
                        print("+==================+")
                        break
                    curr_obs = env.step(step_actions)

    except KeyboardInterrupt:  # stop running with a keyboard interrupt
        pass
    print("Program complete.")
    print("")


# ------ execution section ------
if __name__ == "__main__":
    app.run(main)
