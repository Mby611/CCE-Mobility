# Third-Party Materials

This repository publishes original CCE-Mobility methods, adapters, synthetic fixtures and paper figures. It does not bundle upstream code, original datasets, environment databases, replay sandbox files or raw experimental traces.

| Material | Official source | Treatment |
| --- | --- | --- |
| TravelPlanner code | https://github.com/OSU-NLP-Group/TravelPlanner | Obtain separately; preserve the upstream MIT notice. |
| TravelPlanner data | https://huggingface.co/datasets/osunlp/TravelPlanner | Obtain separately; follow the dataset's CC BY 4.0 attribution terms. |
| MobilityBench code | https://github.com/AMAP-ML/MobilityBench | Obtain separately; consult the license at the pinned upstream revision. |
| MobilityBench data and sandbox | https://huggingface.co/datasets/GD-ML/MobilityBench | Obtain separately; consult dataset/source terms, independently of our code license. |
| SkillOpt | https://github.com/microsoft/SkillOpt | Optional upstream trainer; preserve its MIT notice. |
| Reflexion | https://github.com/noahshinn/reflexion | Cited method; our benchmark-specific adaptation is not a copy of its complete runtime. |
| Voyager | https://github.com/MineDojo/Voyager | Cited method; our benchmark-specific adaptation is not its Minecraft runtime. |

The MIT License applies to original code in this repository, not to downloaded third-party inputs. Prepared source and dataset revisions are recorded per experiment. Do not redistribute official data or sandbox files based solely on our license.

TP evaluator compatibility changes the local query-loading interface and subset denominators while retaining upstream constraint functions. MB compatibility code controls explicit inference parameters and runtime lifetimes. See the reproduction guides for scope and recorded hashes.
