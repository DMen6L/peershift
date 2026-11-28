"""Best-effort Git identity plus a fingerprint of the executing Python sources."""
import hashlib
from pathlib import Path
import subprocess


def code_provenance(source_root=None):
    """Anchor Git queries to the package, never to an unrelated working directory.

    Dirty covers tracked/staged changes and nonignored untracked files across
    that repository. None means unavailable, not clean. The source hash covers
    relative names and bytes of package .py files, independently of Git.
    """
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    root = root.resolve()
    result = dict(code_revision='unavailable', code_worktree_dirty=None, code_source_sha256=None)
    try:
        paths = sorted(root.rglob('*.py'))
        if paths:
            digest = hashlib.sha256(b'peershift-python-source-v1\0')
            for path in paths:
                name, content = path.relative_to(root).as_posix().encode(), path.read_bytes()
                for value in (name, content):
                    digest.update(len(value).to_bytes(8, 'big'))
                    digest.update(value)
            result['code_source_sha256'] = digest.hexdigest()
    except OSError:
        pass
    try:
        def git(*args):
            return subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args], capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
        result['code_revision'] = git('rev-parse', 'HEAD')
        result['code_worktree_dirty'] = bool(git('status', '--porcelain', '--untracked-files=normal'))
    except (OSError, subprocess.SubprocessError):
        pass
    return result
