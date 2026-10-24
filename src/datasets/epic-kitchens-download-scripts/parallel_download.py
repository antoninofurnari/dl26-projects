"""
Download parallelo dei video EPIC-KITCHENS (riusa URL/errata/filtri di epic_downloader.py).

Il server di Bristol limita la banda per singola connessione (~0.28 MB/s),
quindi ogni video viene diviso in pezzi (default 128 MB) scaricati in parallelo
con richieste HTTP Range: il throughput totale e' ~0.28 MB/s x numero di worker,
e anche i file piu' grandi (8 GB) non restano appesi a una sola connessione.

Uso:
    python3 parallel_download.py --workers 32 --participants P22
    python3 parallel_download.py --output-path /Volumes/SSD --workers 32 --participants P01,P02

- I pezzi stanno in <video>.MP4.chunks/ e vengono riuniti appena il video e' completo
- Ogni video riunito viene verificato con l'md5 di data/md5.csv
- Riusa i .part lasciati dalla versione precedente (o da epic_downloader.py)
- Salta i file gia' completi (dimensione == Content-Length)
- Puo' essere interrotto (Ctrl+C) e rilanciato in qualsiasi momento: riprende da dove era
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from epic_downloader import EpicDownloader

READ = 1024 * 1024


def collect_jobs(output_path, participants, challenges):
    """Fa girare EpicDownloader in modalita' 'dry run' per raccogliere (url, output_path) e la tabella md5."""
    jobs = []
    d = EpicDownloader(base_output=output_path)
    d.download_file = lambda url, out: jobs.append((url, out))
    d.file_already_downloaded = lambda *a: False
    d.download(what=('videos',), participants=participants, challenges=challenges)
    md5 = {}
    for url, _ in jobs:
        for base, v in ((d.base_url_100, '100'), (d.base_url_55, '55')):
            key = url[len(base) + 1:] if url.startswith(base + '/') else None
            if key in d.md5[v]:
                md5[url] = d.md5[v][key]
    return jobs, md5


def remote_size(url, retries=10):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers['Content-Length'])
        except Exception as e:
            print(f'[retry HEAD {attempt + 1}] {url.rsplit("/", 1)[-1]}: {e}', flush=True)
            time.sleep(30)
    raise RuntimeError(f'impossibile leggere la dimensione di {url}')


def md5_checksum(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(16 * READ):
            h.update(chunk)
    return h.hexdigest()


def plan_chunks(out, total, chunk_size):
    """Divide il file in intervalli [start, end) e li salva in un manifest, cosi' restano stabili tra i rilanci."""
    cdir = out + '.chunks'
    manifest = os.path.join(cdir, 'manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as f:
            m = json.load(f)
        if m['total'] == total:
            return cdir, [tuple(r) for r in m['ranges']]
        for name in os.listdir(cdir):  # il file remoto e' cambiato: i pezzi vecchi non valgono piu'
            os.remove(os.path.join(cdir, name))
    os.makedirs(cdir, exist_ok=True)
    ranges, start = [], 0
    legacy = out + '.part'
    if os.path.exists(out) and not os.path.exists(legacy):  # file parziale lasciato da epic_downloader.py
        os.replace(out, legacy)
    if os.path.exists(legacy):  # il .part contiguo da 0 diventa il primo pezzo, gia' completo
        start = min(os.path.getsize(legacy), total)
        os.replace(legacy, os.path.join(cdir, '0000'))
        if start:
            ranges.append((0, start))
    while start < total:
        ranges.append((start, min(start + chunk_size, total)))
        start += chunk_size
    with open(manifest, 'w') as f:
        json.dump({'total': total, 'ranges': ranges}, f)
    return cdir, ranges


def download_chunk(url, path, start, end, retries=50):
    for attempt in range(retries):
        try:
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have > end - start:  # pezzo corrotto: ricomincia
                os.remove(path)
                have = 0
            if have == end - start:
                return True
            req = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end - 1}'})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, 'ab') as f:
                if r.status != 206:
                    raise RuntimeError(f'il server ha ignorato il Range (status {r.status})')
                while chunk := r.read(READ):
                    f.write(chunk)
            if os.path.getsize(path) == end - start:
                return True
        except Exception as e:
            print(f'[retry {attempt + 1}] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: {e}', flush=True)
            time.sleep(min(30 * (attempt + 1), 300))
    return False


def assemble(out, cdir, ranges):
    """Riunisce i pezzi in ordine cancellando ognuno appena accodato (picco di disco = un pezzo)."""
    tmp = out + '.assembling'
    size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    ends = [0] + [e for _, e in ranges]
    if size not in ends:  # interruzione a meta' di un pezzo: torna all'ultimo confine completo
        size = max(e for e in ends if e <= size)
        with open(tmp, 'ab') as f:
            f.truncate(size)
    with open(tmp, 'ab') as f:
        for i, (s, e) in enumerate(ranges):
            part = os.path.join(cdir, f'{i:04d}')
            if e <= size:  # gia' accodato in un'esecuzione precedente
                if os.path.exists(part):
                    os.remove(part)
                continue
            with open(part, 'rb') as p:
                while chunk := p.read(16 * READ):
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
            os.remove(part)
    os.replace(tmp, out)
    for name in os.listdir(cdir):
        os.remove(os.path.join(cdir, name))
    os.rmdir(cdir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-path', default=str(Path.home()))
    p.add_argument('--workers', type=int, default=32, help='connessioni simultanee totali')
    p.add_argument('--chunk-mb', type=int, default=128)
    p.add_argument('--participants', default='all', help='es. P01,P02 (default: all)')
    p.add_argument('--challenges', default='cmr', help='cmr = action retrieval (default)')
    args = p.parse_args()

    participants = 'all' if args.participants == 'all' else args.participants.split(',')
    jobs, md5 = collect_jobs(args.output_path, participants, args.challenges.split(','))
    print(f'{len(jobs)} video da controllare/scaricare', flush=True)

    files, tasks, skipped = {}, [], 0
    for url, out in jobs:
        total = remote_size(url)
        if os.path.exists(out) and os.path.getsize(out) == total:
            skipped += 1
            continue
        cdir, ranges = plan_chunks(out, total, args.chunk_mb * 1024 * 1024)
        files[out] = {'url': url, 'cdir': cdir, 'ranges': ranges, 'left': len(ranges), 'total': total}
        tasks += [(out, i) for i in range(len(ranges))]
    have = lambda path: os.path.getsize(path) if os.path.exists(path) else 0
    todo = sum(e - s - have(os.path.join(f['cdir'], f'{i:04d}'))
               for f in files.values() for i, (s, e) in enumerate(f['ranges']))
    print(f'{skipped} gia\' completi, {len(files)} da scaricare ({todo / 1e9:.1f} GB mancanti) '
          f'in {len(tasks)} pezzi con {args.workers} connessioni', flush=True)

    lock, t0, done_bytes, failed, finished, n = threading.Lock(), time.time(), 0, [], 0, 0

    def run(out, i):
        f = files[out]
        s, e = f['ranges'][i]
        path = os.path.join(f['cdir'], f'{i:04d}')
        before = have(path)
        ok = download_chunk(f['url'], path, s, e)
        new = (e - s - before) if ok else 0
        with lock:
            f['left'] -= 1
            last = f['left'] == 0
            if not ok:
                f['failed'] = True
        if last and not f.get('failed'):
            assemble(out, f['cdir'], f['ranges'])
            expected = md5.get(f['url'])
            if expected and md5_checksum(out) != expected:
                os.replace(out, out + '.badmd5')
                return out, 'BADMD5', new
            return out, 'video ok', new
        return out, 'pezzo ok' if ok else 'FAILED', new

    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(run, out, i) for out, i in tasks]
        for fut in as_completed(futures):
            out, status, size = fut.result()
            done_bytes += size
            n += 1
            if status in ('FAILED', 'BADMD5'):
                failed.append(f'{status} {out}')
            if status == 'video ok':
                finished += 1
            if status != 'pezzo ok' or n % 20 == 0:
                el = time.time() - t0
                rate = done_bytes / max(el, 1) / 1e6
                eta = (todo - done_bytes) / max(rate * 1e6, 1) / 3600
                print(f'{status:9} {os.path.basename(out):14} | video finiti {finished}/{len(files)} | '
                      f'{done_bytes / 1e9:.1f}/{todo / 1e9:.1f} GB | {rate:.2f} MB/s | ETA {eta:.1f} h', flush=True)

    if failed:
        print('Problemi (rilancia lo script; i .badmd5 vanno cancellati prima):', *failed, sep='\n  ')
        sys.exit(1)
    print(f'Fatto in {(time.time() - t0) / 3600:.1f} h, tutti i video verificati con md5.')


if __name__ == '__main__':
    main()
