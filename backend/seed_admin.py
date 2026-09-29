"""Interactively create or reset a local admin/facilitator account."""

import argparse
import getpass

from backend import db
from backend.auth import hash_password


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("admin", "facilitator"), required=True)
    parser.add_argument("--email", help="Account email (defaults to the development seed account)")
    args = parser.parse_args()
    email = (args.email or f"{args.role}@test.local").strip().lower()
    password = getpass.getpass("Password (at least 12 characters): ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        parser.error("Passwords do not match.")
    if len(password) < 12 or len(password.encode("utf-8")) > 72:
        parser.error("Password must be at least 12 characters and no more than 72 UTF-8 bytes.")
    db.init_db()
    user = db.update_user_credentials(email, hash_password(password), args.role)
    print(f"Seeded {user['role']} account: {user['email']}")


if __name__ == "__main__":
    main()
