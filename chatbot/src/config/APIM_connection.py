
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.core.credentials import AccessToken
import logging

logger = logging.getLogger(__name__)


class ApimSubscriptionPolicy(SansIOHTTPPolicy):
    """Custom policy for APIM Subscription Key Authentication"""
    def __init__(self, subscription_key: str):
        super().__init__()
        self.subscription_key = subscription_key.strip() if subscription_key else ""
        self.operation_id = None 

    def on_request(self, request):
        """Add Ocp-Apim-Subscription-Key header to all requests"""
        if not self.subscription_key:
            logger.warning("APIM subscription key is empty")
            return
    
        if "authorization" in request.http_request.headers:
            del request.http_request.headers["authorization"]
        
        request.http_request.headers["Ocp-Apim-Subscription-Key"] = self.subscription_key
        logger.debug(f"Added APIM header to request: {request.http_request.method} {request.http_request.url[:100]}")
        
    def on_response(self, request, response):
        """Extract and store operation ID from Operation-Location header"""
        if response.http_response.status_code >= 400:
            logger.error(f"APIM Response Error: {response.http_response.status_code}")
            if hasattr(response.http_response, 'text'):
                response_text = response.http_response.text() if callable(response.http_response.text) else response.http_response.text
                logger.error(f"Response body: {response_text[:500]}") if response_text else None

class NoOpCredential:
    """Dummy credential for APIM-based authentication"""
    def get_token(self, *args, **kwargs):
        from .appSettings import get_settings
        api_key = get_settings().API_KEY
        return AccessToken(api_key, 0)
