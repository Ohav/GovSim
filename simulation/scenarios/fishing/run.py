import os

import numpy as np
from omegaconf import DictConfig, OmegaConf

from simulation.persona import EmbeddingModel
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .environment import FishingConcurrentEnv, FishingPerturbationEnv

# Added for interventions
from simulation.scenarios.common.environment.concurrent_env import get_discussion_day
from datetime import datetime, timedelta

def run(
    cfg: DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
    if cfg.agent.agent_package == "persona_v3":
        from .agents.persona_v3 import FishingPersona
        from .agents.persona_v3.cognition import utils as cognition_utils

        if cfg.agent.system_prompt == "v3":
            cognition_utils.SYS_VERSION = "v3"
        elif cfg.agent.system_prompt == "v3_nocom":
            cognition_utils.SYS_VERSION = "v3_nocom"
        else:
            cognition_utils.SYS_VERSION = "v1"
        if cfg.agent.cot_prompt == "think_step_by_step":
            cognition_utils.REASONING = "think_step_by_step"
        elif cfg.agent.cot_prompt == "deep_breath":
            cognition_utils.REASONING = "deep_breath"
    else:
        raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

    personas = {
        f"persona_{i}": FishingPersona(
            cfg.agent,
            wrapper,
            embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
        )
        for i in range(5)
    }

    # NOTE persona characteristics, up to design choices
    num_personas = cfg.personas.num

    identities = {}
    for i in range(num_personas):
        persona_id = f"persona_{i}"
        identities[persona_id] = PersonaIdentity(
            agent_id=persona_id, **cfg.personas[persona_id]
        )

    # Standard setup
    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    threshold = cfg.intervention.threshold
    if cfg.intervention.agent_resets > 0:
        if cfg.intervention.clf == 'None':
            raise Exception(f"Invalid classifier path given: {cfg.intervention.clf}")
        classifier = get_classifier_function(cfg.intervention.clf)
    else:
        classifier = lambda x: 0

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)
        personas[persona].num_resets = cfg.intervention.agent_resets

    for persona in personas:
        for other_persona in personas:
            # also add self reference, for conversation
            personas[persona].add_reference_to_other_persona(personas[other_persona])

    if cfg.env.class_name == "fishing_perturbation_env":
        env = FishingPerturbationEnv(cfg.env, experiment_storage, agent_id_to_name)
    elif cfg.env.class_name == "fishing_perturbation_concurrent_env":
        env = FishingConcurrentEnv(cfg.env, experiment_storage, agent_id_to_name)
    else:
        raise ValueError(f"Unknown environment class: {cfg.env.class_name}")
    agent_id, obs = env.reset()

    collected_features = {'max entropy': [], 'max varentropy': [], 'max kurtosis': []}

    while True:
        agent = personas[agent_id]
        action = agent.loop(obs)

        if (action.conv_features is not None) and (env.phase == env.POOL_LOCATION) and (env.num_round > 0):
            # Attempt intervention
            features = action.conv_features
            if cfg.intervention.type == 'conversation':
                for k in features:
                    collected_features[k].append(features[k])
                
                if len(collected_features['max entropy']) == num_personas:
                    classifier_features = min(collected_features['max entropy']), min(collected_features['max varentropy']), min(collected_features['max kurtosis'])
                    collecteD_features = {k: [] for k in collected_features.keys()}
                    
                    classifier_score = classifier(env.num_round, classifier_features[0], classifier_features[1], classifier_features[2])
                    print(f"Got score: {classifier_score}")
                    if (classifier_score > threshold) and (agent.num_resets > 0):
                        agent.num_resets -= 1
                        # Lower rounds by one
                        env.num_round -= 1

                        print(f"Decided to reset conversation! {agent_id}")
                        for agent_selection in env.agents:
                            env.internal_global_state["next_location"][agent_selection] = "restaurant"
                            
                            # Reset time to the discussion time earlier
                            cur_time = env.internal_global_state["next_time"][agent_selection]
                            last_month = cur_time.replace(month=cur_time.month - 1)
                            env.internal_global_state["next_time"][agent_selection] = (get_discussion_day(last_month))

                            print("Clean last week thoughts")
                            thoughts = list(agent.memory.thought_id_to_node.keys())
                            week_ago = cur_time - timedelta(days=7)
                            for k in thoughts:
                                if (agent.memory.thought_id_to_node[k].created > week_ago):
                                    del agent.memory.thought_id_to_node[k]
                            
                        
                            # Insert "restaurant" obs to first agent, forcing entire group to recreate the conversation
                        first_agent = env._agent_selector.reset()
                        env.agent_selection = first_agent
                        while env.phase != 'restaurant':
                            env.phase = env._phase_selector.next()
                        
                        obs = env._observe_restaurant(first_agent)
                        action = personas[first_agent].loop(obs)
                        print("finished intervention")

                    
            elif cfg.intervention.type == 'reflection':
                if (features['max entropy'] > 1) and (agent.num_resets > 0):
                    print(f"Decided to reset agent! {agent_id}")
                    print(f"Previous action was: {action}")
                    agent.num_resets -= 1
                    # Insert "home" obs to agent, to force it to reflect again
                    home_obs = env._observe_home(agent)
                    agent.loop(home_obs)
                    # Then, recreate the action. Note that loop on fish observation does NOT affect underlying stuff like memory 
                    action = agent.loop(obs)
                    print(f"New action is: {action}!")
                
        (
            agent_id,
            obs,
            rewards,
            termination,
        ) = env.step(action)

        stats = {}
        STATS_KEYS = [
            "conversation_resource_limit",
            *[f"persona_{i}_collected_resource" for i in range(5)],
        ]
        for s in STATS_KEYS:
            if s in action.stats:
                stats[s] = action.stats[s]

        if np.any(list(termination.values())):
            logger.log_game(
                {
                    "num_resource": obs.current_resource_num,
                    **stats,
                },
                last_log=True,
            )
            break
        else:
            logger.log_game(
                {
                    "num_resource": obs.current_resource_num,
                    **stats,
                }
            )

        logger.save(experiment_storage, agent_name_to_id)

    env.save_log()
    for persona in personas:
        personas[persona].memory.save()


def get_classifier_function(classifier_path):
    import sys
    sys.path.append('/home/morg/students/ohavbarbi/multiAgent/')
    import pickle
    import torch
    if classifier_path == "Linear":
        def linear_classifier(turn, entropy, varentropy, kurtosis):
            norm_params = (6.321957870168004e-09, 1.105892456367376)
            entropy = (entropy - norm_params[0]) / (norm_params[1] - norm_params[0])
            return entropy
        return linear_classifier
    
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)

    def run_classifier(turn, entropy, varentropy, kurtosis):
        features = torch.Tensor([turn, entropy, varentropy, kurtosis]).reshape(1, -1)
        return classifier(features)
    return run_classifier