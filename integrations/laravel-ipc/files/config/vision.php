<?php

/*
 * Vision Service (OCR) integration — see integrations/laravel-ipc/README.md in the
 * vision-service repo. Vision Service only returns what it read + confidence + a
 * format flag; the PASS / FAIL / REVIEW decision is made here in Laravel
 * (App\Services\Vision\VisionEvaluator), per the settled IPC architecture.
 */
return [

    'base_url' => env('VISION_SERVICE_URL', 'http://100.100.160.31:8000'),

    // One OCR request takes ~25s on the pilot VPS and the service processes requests one at
    // a time (no parallelism yet), so a second station waiting in line can take ~50s+.
    'timeout' => (int) env('VISION_SERVICE_TIMEOUT', 90),

    'connect_timeout' => (int) env('VISION_SERVICE_CONNECT_TIMEOUT', 5),

    // field_types this app is allowed to send. Must exist in the Vision Service's
    // emboss_format_patterns.json. Laravel always decides the field_type from the workflow
    // step — Vision Service never guesses it from the photo.
    'field_types' => [
        'tube_exp_date',
        'tube_mfd_date',
        'tube_emboss_default',
    ],

    // How each field_type's OCR value is checked against IPC data. A field_type with no rule
    // here always ends up REVIEW (never PASS) until a rule is added.
    //   batch_exp_date      : OCR'd EXP (DDMMYY) must equal ipc_batches.exp_date
    //   mfd_plus_shelf_life : OCR'd MFD + master_products.shelf_life_months must equal
    //                         ipc_batches.exp_date
    'rules' => [
        'tube_exp_date' => 'batch_exp_date',
        'tube_mfd_date' => 'mfd_plus_shelf_life',
    ],

];
