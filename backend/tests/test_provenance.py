"""Provenance describes the source repository, independent of launch directory."""
import subprocess

from peershift.provenance import code_provenance


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_clean_dirty_staged_untracked_and_source_fingerprint(tmp_path, monkeypatch):
    repo=tmp_path/'repo'
    package=repo/'package'
    package.mkdir(parents=True)
    source=package/'model.py'
    source.write_text('value=1\n')
    git(repo,'init','--quiet')
    git(repo,'add','.')
    git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','-c','commit.gpgsign=false','-c','core.hooksPath=/dev/null','commit','--quiet','-m','fixture')
    monkeypatch.chdir(tmp_path)
    clean=code_provenance(package)
    assert clean['code_revision']==git(repo,'rev-parse','HEAD')
    assert clean['code_worktree_dirty'] is False
    assert len(clean['code_source_sha256'])==64
    source.write_text('value=2\n')
    modified=code_provenance(package)
    assert modified['code_revision']==clean['code_revision']
    assert modified['code_worktree_dirty'] is True
    assert modified['code_source_sha256']!=clean['code_source_sha256']
    git(repo,'add','.')
    assert code_provenance(package)==modified
    source.write_text('value=1\n')
    git(repo,'add','.')
    (repo/'notes.txt').write_text('untracked\n')
    untracked=code_provenance(package)
    assert untracked['code_worktree_dirty'] is True
    assert untracked['code_source_sha256']==clean['code_source_sha256']
    (package/'new.py').write_text('other=3\n')
    assert code_provenance(package)['code_source_sha256']!=clean['code_source_sha256']


def test_no_repository_or_git_is_unknown_not_clean(tmp_path, monkeypatch):
    (tmp_path/'model.py').write_text('value=1\n')
    result=code_provenance(tmp_path)
    assert result['code_revision']=='unavailable'
    assert result['code_worktree_dirty'] is None
    assert result['code_source_sha256'] is not None
    def missing(*args,**kwargs):
        raise FileNotFoundError('git unavailable')
    monkeypatch.setattr(subprocess,'run',missing)
    assert code_provenance(tmp_path)==result
