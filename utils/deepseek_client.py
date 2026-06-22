import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


class DeepSeekResponseTruncated(RuntimeError):
    """Raised when DeepSeek reports finish_reason=length."""

    truncated = True
    finish_reason = 'length'


def _default_transport(url, headers, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


class DeepSeekClient:
    """Main-process-only DeepSeek client with injectable network transport."""

    def __init__(self, base_url, model, temperature=0.0, max_tokens=1024,
                 timeout=60, use_json_response=True, thinking_type=None,
                 transport=None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.use_json_response = bool(use_json_response)
        self.thinking_type = thinking_type
        self.transport = transport or _default_transport

    @staticmethod
    def build_prompt(report_text):
        return (
            'Analyze the evidence report below and return only one JSON object. '
            'The only allowed explicit feature is capability_match. Use this '
            'exact schema:\\n'
            '{"weights":{"capability_match":0.0},"lambda":0.0,'
            '"clip_range":[-2.0,2.0],"rationale":'
            '{"main_failure_modes":[],"expected_effect":[]}}\\n\\n'
            'EVIDENCE REPORT\\n'
            '================\\n'
            + report_text)

    def request_bias_config(self, report_path, prompt=None):
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise RuntimeError('DEEPSEEK_API_KEY is not set')
        if prompt is None:
            report_text = Path(report_path).read_text(encoding='utf-8')
            prompt = self.build_prompt(report_text)
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        if self.thinking_type:
            payload['thinking'] = {'type': self.thinking_type}
        if self.use_json_response:
            payload['response_format'] = {'type': 'json_object'}
        response = self.transport(
            self.base_url + '/chat/completions',
            {
                'Authorization': 'Bearer ' + api_key,
                'Content-Type': 'application/json',
            },
            payload,
            self.timeout)
        choice = response['choices'][0]
        finish_reason = choice.get('finish_reason')
        if finish_reason == 'length':
            raise DeepSeekResponseTruncated(
                'DeepSeek response exceeded max_tokens')
        if finish_reason in ('content_filter', 'insufficient_system_resource'):
            raise RuntimeError('DeepSeek finish_reason={}'.format(finish_reason))
        content = choice['message']['content']
        return prompt, response, self._parse_json_content(content)

    @staticmethod
    def _parse_json_content(content):
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError('DeepSeek message content must be JSON text')
        stripped = content.strip()
        fenced = re.match(
            r'^```(?:json)?\s*(.*?)\s*```$', stripped, flags=re.DOTALL)
        if fenced:
            stripped = fenced.group(1)
        return json.loads(stripped)
