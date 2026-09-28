<?php

namespace App\Services\Vision;

use Illuminate\Http\Client\ConnectionException;
use Illuminate\Support\Facades\Http;

/**
 * Real VisionClient: POST {vision.base_url}/api/analyze as multipart (file + field_type +
 * request_id). debug/reference_date are deliberately NOT sent — debug adds large base64
 * crops to the response, and the reference-date comparison is done by VisionEvaluator in
 * Laravel, not by the Vision Service.
 */
class HttpVisionClient implements VisionClient
{
    public function analyze(string $imageContents, string $filename, string $fieldType, string $requestId): VisionResult
    {
        try {
            $response = Http::baseUrl(rtrim(config('vision.base_url'), '/'))
                ->timeout(config('vision.timeout'))
                ->connectTimeout(config('vision.connect_timeout'))
                ->acceptJson()
                ->attach('file', $imageContents, $filename)
                ->post('/api/analyze', [
                    'field_type' => $fieldType,
                    'request_id' => $requestId,
                ]);
        } catch (ConnectionException $e) {
            return VisionResult::error('VISION_UNREACHABLE: '.$e->getMessage());
        }

        if (! $response->successful()) {
            return VisionResult::error('VISION_HTTP_'.$response->status());
        }

        $data = $response->json();

        if (! is_array($data)) {
            return VisionResult::error('VISION_INVALID_JSON');
        }

        // main.py returns {"error": "..."} (no status key) when the image can't be decoded.
        if (isset($data['error']) && ! isset($data['status'])) {
            return VisionResult::error('VISION_ERROR: '.$data['error']);
        }

        return VisionResult::fromResponse($data);
    }
}
