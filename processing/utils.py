import os


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set.

    Copied from XCP-D.
    """
    import subprocess

    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise RuntimeError(
            f'Non zero return code: {process.returncode}\n{command}\n\n{process.stdout.read()}'
        )


def to_bidsuri(filename, dataset_dir, dataset_name):
    return f'bids:{dataset_name}:{os.path.relpath(filename, dataset_dir)}'


def get_filename(name_source, layout, entities, dismiss_entities=None):
    if dismiss_entities is None:
        dismiss_entities = []

    source_entities = layout.get_file(name_source).get_entities()
    source_entities = {k: v for k, v in source_entities.items() if k not in dismiss_entities}
    entities = {**source_entities, **entities}
    return layout.build_path(**entities)
