# Code provenance and acknowledgements

Renaming project folders does not change the provenance of the code in them.

## Representation learning

`joint_embedding_learning` contains the project's three-sensor encoders, decoders, paired-data pipeline, and alignment training. Its public history starts at `3385705d` and includes work by the TactX coauthors. Commit authorship records who submitted changes; it does not establish that every line was independently authored.

The retained implementation imports `get_resnet` and `NT_Xent` from [Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR). The dependency is pinned to commit `2738efae85b3ca1b70b7919c5d7a7837d9d75235` in `requirements.txt`; its [MIT license](https://github.com/Spijkervet/SimCLR/blob/2738efae85b3ca1b70b7919c5d7a7837d9d75235/LICENSE) belongs to that project. ResNet itself is provided by torchvision.

The previous source tree also contained Bubbles/GelSlim dataset code and a LARS implementation explicitly described as copied from Barlow Twins. Those unused paths have been removed from this branch. The available local history does not identify an external source for every legacy function, so this audit does not claim that the original directory was entirely original code.

## ACT policy learning

`policy_learning` is the renamed downstream policy package. Its original package metadata credits **Wang Lab**; that attribution is preserved in `pyproject.toml`. This is an adapted policy implementation, not an independently authored replacement for the upstream policy code.

The policy builds on [ACT](https://github.com/tonyzhaozh/act) and includes adapted [DETR](https://github.com/facebookresearch/detr) components. The existing Facebook copyright headers and [Apache-2.0 license](policy_learning/tactile_policy/third_party/detr/LICENSE) are preserved. This cleanup modifies the training entry point, packaging, device handling, and selected runtime bugs while retaining the policy architecture.

The unused PI0/OpenPI copies and unrelated hardware utilities are removed. Their removal and the folder rename do not imply ownership of any upstream implementation. No repository-wide license is assigned by this cleanup; the bundled license applies to its stated third-party scope.
