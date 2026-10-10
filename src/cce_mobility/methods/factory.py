from cce_mobility.methods.baselines import (
    AgentSkillOnlyMethod,
    OfficialSkillOptArtifactMethod,
    ReflexionLikeMethod,
    VoyagerLikeMethod,
    WMSpecOnlyMethod,
)
from cce_mobility.methods.no_evolution import NoEvolutionMethod
from cce_mobility.methods.skill_evolution import (
    OntologyDiscoveryPhaseSwitchingCoEvolutionMethod,
    SkillEvolutionMethod,
)


def create_method(name: str):
    methods = {
        "no_evolution": NoEvolutionMethod,
        "official_posthoc_no_evolution": NoEvolutionMethod,
        "cce": OntologyDiscoveryPhaseSwitchingCoEvolutionMethod,
        "ontology_phase_switching": OntologyDiscoveryPhaseSwitchingCoEvolutionMethod,
        "skill_evolution": SkillEvolutionMethod,
        "agent_skill_only": AgentSkillOnlyMethod,
        "agent_only": AgentSkillOnlyMethod,
        "wm_spec_only": WMSpecOnlyMethod,
        "official_skillopt": OfficialSkillOptArtifactMethod,
        "reflexion_like": ReflexionLikeMethod,
        "voyager_like": VoyagerLikeMethod,
    }
    if name.lower() not in methods:
        raise ValueError(f"Unknown method: {name}; choose a published method or named ablation")
    return methods[name.lower()]()
