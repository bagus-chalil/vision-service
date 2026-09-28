<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

// Audit trail of every Vision Service call: what was sent, what the service read (as-is),
// and what Laravel decided + why. request_id is the same UUID sent to the Vision Service, so
// a row can be matched against the service's own logs/fallback_cases.jsonl.
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('ipc_logs', function (Blueprint $table) {
            $table->id();
            $table->uuid('request_id')->unique();
            $table->foreignId('ipc_batch_id')->constrained('ipc_batches');
            $table->string('stage');
            $table->string('field_type');
            $table->string('image_path');

            $table->string('vision_status');
            $table->string('engine_used')->nullable();
            $table->decimal('confidence', 5, 4)->nullable();
            $table->boolean('format_valid')->nullable();
            $table->string('ocr_value')->nullable();
            $table->text('raw_ocr_text')->nullable();
            $table->text('error_reason')->nullable();
            $table->string('match_method')->nullable();
            $table->unsignedInteger('processing_time_ms')->nullable();
            $table->json('vision_response')->nullable();

            $table->string('decision');
            $table->text('decision_reason');
            $table->string('expected_value')->nullable();
            $table->string('computed_value')->nullable();

            $table->foreignId('created_by')->constrained('users');
            $table->timestamps();

            $table->index(['ipc_batch_id', 'stage', 'field_type']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('ipc_logs');
    }
};
