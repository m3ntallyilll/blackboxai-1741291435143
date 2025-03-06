import http.server
import socketserver
import webbrowser
from threading import Timer

PORT = 8000

def open_browser():
    webbrowser.open(f'http://localhost:{PORT}')

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    Timer(1, open_browser).start()
    httpd.serve_forever()
