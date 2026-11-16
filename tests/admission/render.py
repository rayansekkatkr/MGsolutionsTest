"""Extract what ECS will run for the two proxy containers, straight from exam/template.yaml.

Usage: python render.py TEMPLATE IMAGES_JSON OUT_DIR
Writes, with parameter defaults substituted, haproxy.cfg, command.txt and user.txt for the proxy
of a GPU task, edge.cfg, edge_command.txt and edge_user.txt for an edge proxy, and image.txt,
max_batch.txt, max_waiting.txt, workers.txt, so the test exercises the template itself and not a copy of its
configuration. The one thing replaced is the fleet ALB's DNS name in the edge configuration,
which becomes FLEET_ALB (to be set to the address of the stand-in used in the test).
"""
import json
import pathlib
import sys

import yaml


class Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return {suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {suffix: loader.construct_sequence(node, deep=True)}
    return {suffix: loader.construct_mapping(node, deep=True)}


Loader.add_multi_constructor("!", _tag)

template, images, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
doc = yaml.load(open(template), Loader)
defaults = {name: spec.get("Default") for name, spec in doc["Parameters"].items()}
containers = doc["Resources"]["Task"]["Properties"]["ContainerDefinitions"]
proxy = next(c for c in containers if c["Name"] == "proxy")

config = next(e for e in proxy["Environment"] if e["Name"] == "HAPROXY_CFG")["Value"]["Sub"]
for name, value in defaults.items():
    if value is not None:
        config = config.replace("${%s}" % name, str(value))
assert "${" not in config, "unresolved substitution in the HAProxy configuration"
assert proxy["EntryPoint"] == ["sh", "-c"] and len(proxy["Command"]) == 1

image_param = proxy["Image"]["Ref"]
image = json.load(open(images))[image_param[len("Image"):]]

out.mkdir(parents=True, exist_ok=True)
(out / "haproxy.cfg").write_text(config)
(out / "command.txt").write_text(proxy["Command"][0])
(out / "user.txt").write_text(str(proxy["User"]))
(out / "image.txt").write_text(image)
(out / "max_batch.txt").write_text(str(defaults["MaxBatch"]))
(out / "max_waiting.txt").write_text(str(defaults["MaxWaiting"]))
(out / "workers.txt").write_text(str(defaults["Workers"]))

edge = doc["Resources"]["EdgeTask"]["Properties"]["ContainerDefinitions"][0]
edge_config = next(e for e in edge["Environment"] if e["Name"] == "HAPROXY_CFG")["Value"]["Sub"]
for name, value in defaults.items():
    if value is not None:
        edge_config = edge_config.replace("${%s}" % name, str(value))
# In the test the fleet ALB is a stand-in at a fixed address: no DNS name to follow.
lines = edge_config.splitlines()
fleet = [i for i, line in enumerate(lines) if "${FleetAlb.DNSName}" in line]
assert len(fleet) == 1 and lines[fleet[0]].strip().startswith("server-template alb")
lines[fleet[0]] = "    server alb FLEET_ALB"
edge_config = "\n".join(lines) + "\n"
assert "${" not in edge_config, "unresolved substitution in the edge configuration"
assert edge["EntryPoint"] == ["sh", "-c"] and len(edge["Command"]) == 1 and edge["Image"]["Ref"] == image_param
(out / "edge.cfg").write_text(edge_config)
(out / "edge_command.txt").write_text(edge["Command"][0])
(out / "edge_user.txt").write_text(str(edge["User"]))
print(f"image={image} task proxy: user={proxy['User']} arrival_rate={defaults['MaxArrivalRate']}/s in_system={defaults['MaxInSystem']}"
      f" | edge proxy: user={edge['User']} rate={defaults['EdgeRate']}/s")
