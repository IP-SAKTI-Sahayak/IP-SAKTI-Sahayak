"""Check registered source URLs before a demo: python -m backend.check_sources."""

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.source_registry import TRUSTED_SOURCES


def check_url(url):
    request = Request(url, headers={"Range": "bytes=0-0", "User-Agent": "IP-SAKTI-Sahayak-source-check/1.0"})
    try:
        with urlopen(request, timeout=12) as response:
            return response.status, None
    except HTTPError as error:
        return error.code, str(error.reason)
    except (URLError, TimeoutError, OSError) as error:
        return None, str(error)


def main():
    failures = 0
    checked = 0
    for document, metadata in TRUSTED_SOURCES.items():
        for field in ("official_pdf_url", "official_listing_url"):
            url = metadata.get(field)
            if not url:
                continue
            checked += 1
            status, error = check_url(url)
            if status is not None and 200 <= status < 400:
                print(f"OK {status} | {document} | {field} | {url}")
            else:
                failures += 1
                detail = f"HTTP {status}" if status is not None else "REQUEST FAILED"
                if error:
                    detail += f" ({error})"
                print(f"ERROR {detail} | {document} | {field} | {url}")
    print(f"Checked {checked} URL(s); {failures} error(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
