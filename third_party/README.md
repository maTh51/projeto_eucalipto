# third_party

This directory is the default location for external dependencies managed as git submodules.

Initialize after cloning:

```bash
git submodule update --init --recursive
```

The runner also supports `external_paths` in the pipeline config for non-submodule layouts.
