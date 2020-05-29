import pkg_resources as pkg_res
import sys, os, re, pip, shutil
import os.path as P


__wd = '3rdparty'  # working dir == `pwd`/3rdparty
__offline = {}

def load_offline_json(path: str):
    import json
    '''
        {file_name}: {
            'url': {url},
            'version': {version_string},
            'md5': {md5}
        }
    '''
    with open(path, "r") as f:
        ret = json.load(f)
    assert isinstance(ret, dict)
    need_key = ['url', 'version', 'md5']
    md5_pat = re.compile(r'^[0-9a-f]{32}$')
    for k in ret:
        assert isinstance(k, str) \
            and '\x00' not in k \
            and '\\' not in k \
            and '/' not in k
        for need_k in need_key:
            assert need_k in ret[k] \
                and isinstance(ret[k][need_k], str)
        assert md5_pat.match(ret[k]['md5']) is not None
    return ret


def parse_version(ver_str: str):
    parts = ver_str.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else None
    patch = int(parts[2]) if len(parts) > 2 else None
    return major, minor, patch


def download(url: str, dest: str):
    from requests import get
    import subprocess
    print("downloading " + dest + " from\n    " + url, file=sys.stderr)
    # offline mode
    if len(__offline) > 0:
        file_name = P.basename(dest)
        assert url == __offline[file_name]['url']
        fin = subprocess.run(['md5sum', dest], stdout=subprocess.PIPE)
        assert __offline[file_name]['md5'] == \
            fin.stdout.decode('utf-8').split(" ")[0]
        return __offline[file_name]['version']
    proxies=dict()
    if "PROXY" in os.environ.keys():
        proxy_url = os.environ["PROXY"]
        proxies = { "http": proxy_url, "https": proxy_url, "ftp": proxy_url }
    # online mode
    with open(dest, "wb") as f:
        response = get(url, proxies=proxies)
        f.write(response.content)
        return None


def extract_tar(tar: str, pattern: str, dest_dir: str):
    import tarfile
    pat = re.compile(pattern)
    count = 0
    with tarfile.open(tar, "r") as t:
        for f in t.getmembers():
            if len(pat.findall(f.name)) > 0:
                t.extract(f, dest_dir)
                count += 1
    return count


def detect_protobuf(pkgs: dict):
    if 'protobuf' not in pkgs:
        raise Exception("protobuf not found")
    protobuf = pkgs['protobuf']
    if P.exists(P.join(__wd, 'protobuf')):
        return

    major, minor, patch = parse_version(protobuf.version)
    if major != 3 or minor < 5 or patch is None:
        raise Exception("Incompatible protobuf version " + protobuf.version)

    protobuf_pkg = P.join(
        __wd,
        'protobuf-%d.%d.%d.tar.gz' % (major, minor, patch)
    )
    url = 'https://github.com/protocolbuffers/protobuf/archive/' + \
        'v%d.%d.%d.tar.gz' % (major, minor, patch)
    got_ver = download(url, protobuf_pkg)
    if got_ver is not None and got_ver != protobuf.version:
        raise Exception(
            "Offline package with invalid version: expect %s, got %s" %
            (protobuf.version, got_ver)
        )

    extract_tar(protobuf_pkg, r'.*', __wd)

    haeder_list_f = None
    protobuf_dir = None
    for f in os.listdir(__wd):
        if 'protobuf' in f and P.isdir(P.join(__wd, f)):
            protobuf_dir = P.join(__wd, f)
            break

    protoc_build_dir = protobuf_dir + '/static_build'
    os.mkdir(protoc_build_dir)
    if 0 != os.system('sh {}/build_protoc.sh {}'
                      .format(P.dirname(__file__),
                              protoc_build_dir)):
        raise RuntimeError('Fail to build protoc')

    shutil.rmtree(protobuf_dir)
                
    library = []
    for d, sub_d, sub_f in os.walk(P.join(protobuf.location, 'google')):
        for f in sub_f:
            if f.endswith('.so'):
                library.append(P.join(d, f))
    print('protobuf==' + protobuf.version)
    return library


def detect_dependency():
    pkgs = {p.project_name: p for p in pkg_res.working_set}
    os.makedirs(__wd, exist_ok=True)
    
    if P.exists(P.join(__wd, 'offline.json')):
        print("Use offline mode, loading " + P.join(__wd, 'offline.json'))
        __offline.update(load_offline_json(P.join(__wd, 'offline.json')))
        print("offline.json loaded with " + str(len(__offline)) + " archieve")

    detect_protobuf(pkgs)

if __name__ == '__main__':
    detect_dependency()
