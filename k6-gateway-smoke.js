import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  vus: 5,
  duration: '1m',
};

export default function () {
  const url = 'http://127.0.0.1:9910/answer';
  const payload = JSON.stringify({ query: 'smoke test', top_k: 2 });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'x-admin-token': '',
    },
  };
  const res = http.post(url, payload, params);
  sleep(1);
}
