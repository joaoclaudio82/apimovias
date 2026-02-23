from pathlib import Path
from typing import Dict, List
import logging

import httpx

from api.settings import settings
from api.utils.path_utils import PathUtils

logger = logging.getLogger(__name__)


class AiIntegrationService:
    @staticmethod
    def _parse_targets() -> List[str]:
        return [t.strip() for t in settings.AI_SYNC_TARGETS.split(',') if t.strip()]

    @staticmethod
    def _request_service_token(client: httpx.Client) -> str:
        response = client.post(
            '/auth/service_token',
            json={
                'client_id': settings.AI_SERVICE_CLIENT_ID,
                'client_secret': settings.AI_SERVICE_CLIENT_SECRET,
            },
        )
        response.raise_for_status()

        payload = response.json()
        token = payload.get('access_token')
        if not token:
            raise RuntimeError("Resposta do módulo AI sem 'access_token'.")
        return token

    @staticmethod
    def _sync_target_from_shared_path(
        client: httpx.Client,
        token: str,
        target: str,
        csv_path: str,
    ) -> Dict[str, int]:
        response = client.post(
            '/vehicle-profiles/update-csv-path',
            headers={'Authorization': f'Bearer {token}'},
            json={'target': target, 'csv_path': csv_path},
        )
        response.raise_for_status()

        payload = response.json()
        return {
            'n_vehicles': int(payload.get('n_vehicles', 0)),
        }

    @staticmethod
    def sync_profiles_from_csv(path: Path = settings.CSV_PATH) -> Dict[str, object]:
        if PathUtils.is_empty(path):
            raise ValueError(f'CSV para sync AI ausente ou vazio: {path}')

        targets = AiIntegrationService._parse_targets()
        if not targets:
            raise ValueError('AI_SYNC_TARGETS vazio.')

        timeout = max(1, int(settings.AI_HTTP_TIMEOUT_SECONDS))

        results: Dict[str, Dict[str, int]] = {}
        with httpx.Client(base_url=settings.AI_API_BASE_URL.rstrip('/'), timeout=timeout) as client:
            token = AiIntegrationService._request_service_token(client)
            for target in targets:
                results[target] = AiIntegrationService._sync_target_from_shared_path(
                    client=client,
                    token=token,
                    target=target,
                    csv_path=settings.AI_SHARED_CSV_PATH,
                )

        logger.info(
            'Sync AI via volume compartilhado concluído | targets=%s | csv_path=%s',
            ','.join(results.keys()),
            settings.AI_SHARED_CSV_PATH,
        )
        return {
            'mode': 'shared-volume',
            'csv_path': settings.AI_SHARED_CSV_PATH,
            'targets': results,
        }
