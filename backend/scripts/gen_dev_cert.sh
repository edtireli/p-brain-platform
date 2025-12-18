#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p certs

KEY_FILE="certs/dev.key"
CERT_FILE="certs/dev.crt"
CONF_FILE="certs/openssl.cnf"

if [[ -f "$KEY_FILE" && -f "$CERT_FILE" ]]; then
  exit 0
fi

cat >"$CONF_FILE" <<'EOF'
[ req ]
default_bits       = 2048
distinguished_name = req_distinguished_name
x509_extensions    = v3_req
prompt             = no

[ req_distinguished_name ]
CN = 127.0.0.1

[ v3_req ]
subjectAltName = @alt_names
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[ alt_names ]
IP.1 = 127.0.0.1
DNS.1 = localhost
EOF

openssl req \
  -x509 \
  -nodes \
  -newkey rsa:2048 \
  -days 365 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -config "$CONF_FILE"

echo "Generated dev TLS cert at: $CERT_FILE"
echo "If your browser blocks requests, visit https://127.0.0.1:8787/health once and accept the certificate warning."
