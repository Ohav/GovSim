from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from simulation.interventions import feature_utils

from .utils import (
    consider_identity_persona_prompt,
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)


def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
):
    lm = model.start_chain(identity.name, "fishing_cognition_act", "choose_act_options")

    with user():
        system_prompt = get_sytem_prompt(identity)
        system_prompt = system_prompt.replace('100', str(identity.max_resource_num)).replace('<X>', str(identity.max_resource_num - 10)).replace('<X-30>', str(identity.max_resource_num - 40))

        lm += f"{system_prompt}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\n"
        lm += f"Task: With a fishing range set between {interval[0]}-{interval[-1]}, how many tons of fish would you catch this month? "
        lm += reasoning_steps_prompt()
        lm += ' Put the final answer after "Answer:", example Answer: N tons.'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
            save_stop_text=True,
        )
        
        ## Intervention features
        logprobs = lm.last_logprobs
        if logprobs is not None:
            raw_collections, _ = feature_utils.get_gen_metrics(logprobs)
            conv_features = dict()
            for feature in raw_collections.keys():
                conv_features[f'max {feature}'] = max(raw_collections[feature])
        else:
            conv_features = None

        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            stop_regex=f"tons",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)
    return option, lm.html(), conv_features
