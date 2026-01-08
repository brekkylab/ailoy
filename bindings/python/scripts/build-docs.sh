#!/bin/bash

uv pip compile --group docs --quiet -o /tmp/requirements-docs.txt
uvx --with-requirements /tmp/requirements-docs.txt mkdocs build
rm -f /tmp/requirements-docs.txt
