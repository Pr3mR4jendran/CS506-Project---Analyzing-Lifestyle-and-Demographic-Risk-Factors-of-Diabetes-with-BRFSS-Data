import os
import sys
import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1d4Y8tG21URcQON-GSUhAee4kbtOwtd9T?usp=sharing"
OUTPUT_DIR = "../../Data"

def download_gdrive_folder(folder_url: str, output_dir: str) -> None:
    if not folder_url or "drive.google.com" not in folder_url:
        print("[-] Please set a valid Google Drive folder URL in FOLDER_URL.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"[+] Downloading folder:\n    {folder_url}")
    print(f"[+] Target directory:\n    {os.path.abspath(output_dir)}\n")

    try:
        gdown.download_folder(
            url=folder_url,
            output=output_dir,
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
        print("\n[+] Download completed.")
    except Exception as e:
        print(f"[-] Error while downloading: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_gdrive_folder(FOLDER_URL, OUTPUT_DIR)