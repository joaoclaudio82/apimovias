import os
import urllib

from dotenv import load_dotenv


def configure_proxy(root_dir=None):
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    proxy_config_file = os.path.join(root_dir, '.env.proxy')

    print(proxy_config_file)

    if os.path.exists(proxy_config_file):
        load_dotenv(proxy_config_file)

        use_proxy = os.getenv('USE_PROXY', 'False').lower() == 'true'
        if not use_proxy:
            return

        url_proxy = os.getenv('URL_PROXY')
        user = os.getenv('USER')
        password = os.getenv('PASS')
        no_proxy = os.getenv('NO_PROXY')
        if not all([url_proxy, user, password]):
            raise RuntimeError(
                'USE_PROXY=True mas URL_PROXY, USER ou PASS estão ausentes no .env'
            )

        password_encoded = urllib.parse.quote(password)
        proxy_url1 = f'http://{user}:{password_encoded}@{url_proxy}'
        proxy_url2 = f'https://{user}:{password_encoded}@{url_proxy}'

        os.environ['HTTP_PROXY'] = proxy_url1
        os.environ['HTTPS_PROXY'] = proxy_url2
        os.environ['NO_PROXY'] = no_proxy
        print('Proxy configurado.')
