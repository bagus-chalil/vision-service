<?php

namespace App\Http\Requests;

use App\Models\IpcBatch;
use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Validation\Rule;

class AnalyzeVisionRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'photo' => ['required', 'image', 'mimes:jpeg,jpg,png,webp', 'max:8192'],
            'stage' => ['required', Rule::in([
                IpcBatch::STAGE_STARTUP,
                IpcBatch::STAGE_FILLING,
                IpcBatch::STAGE_PACKING,
                IpcBatch::STAGE_FINISHED,
            ])],
            'field_type' => ['required', Rule::in(config('vision.field_types'))],
        ];
    }
}
