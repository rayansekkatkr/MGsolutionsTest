"""Extract what ECS will run for the proxy container, straight from exam/template.yaml.

Usage: python render.py TEMPLATE IMAGES_JSON OUT_DIR
Writes haproxy.cfg (with parameter defaults substituted), command.txt, user.txt, image.txt and max_batch.txt,
so the test exercises the template itself and not a copy of its configuration.
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
print(f"image={image} user={proxy['User']} arrival_rate={defaults['MaxArrivalRate']}/s in_system={defaults['MaxInSystem']}")
