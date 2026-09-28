<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class IpcLog extends Model
{
    protected $fillable = [
        'request_id',
        'ipc_batch_id',
        'stage',
        'field_type',
        'image_path',
        'vision_status',
        'engine_used',
        'confidence',
        'format_valid',
        'ocr_value',
        'raw_ocr_text',
        'error_reason',
        'match_method',
        'processing_time_ms',
        'vision_response',
        'decision',
        'decision_reason',
        'expected_value',
        'computed_value',
        'created_by',
    ];

    protected function casts(): array
    {
        return [
            'confidence' => 'float',
            'format_valid' => 'boolean',
            'vision_response' => 'array',
        ];
    }

    public function batch()
    {
        return $this->belongsTo(IpcBatch::class, 'ipc_batch_id');
    }

    public function creator()
    {
        return $this->belongsTo(User::class, 'created_by');
    }
}
