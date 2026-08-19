import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


_NOISE_REQUEST_KEY = b"__openpi_request__"
_OBS_KEY = b"obs"
_NOISE_KEY = b"noise"
_RESET_REQUEST_KEY = b"__openpi_reset__"


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_timeout_s: Optional[float] = 10.0,
        recv_timeout_s: Optional[float] = None,
    ) -> None:
        if host in {"0.0.0.0", "::"}:
            logging.warning(
                "Client host %s is a bind address, not a routable destination. Falling back to 127.0.0.1.",
                host,
            )
            host = "127.0.0.1"
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._recv_timeout_s = recv_timeout_s
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=self._connect_timeout_s,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logging.info("Still waiting for server (%s)", exc)
                time.sleep(5)

    @override
    def infer(self, obs: Dict, *, noise: Optional[np.ndarray] = None) -> Dict:  # noqa: UP006
        payload = obs if noise is None else {_NOISE_REQUEST_KEY: {_OBS_KEY: obs, _NOISE_KEY: noise}}
        data = self._packer.pack(payload)
        self._ws.send(data)
        try:
            response = self._ws.recv(timeout=self._recv_timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Timed out waiting for inference response from {self._uri} after {self._recv_timeout_s}s"
            ) from exc
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        self._ws.send(self._packer.pack({_RESET_REQUEST_KEY: True}))
        try:
            response = self._ws.recv(timeout=self._recv_timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Timed out waiting for reset acknowledgement from {self._uri} after {self._recv_timeout_s}s"
            ) from exc
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server during reset:\n{response}")
        # Decode and ignore payload; we only need acknowledgement to keep protocol in sync.
        msgpack_numpy.unpackb(response)
