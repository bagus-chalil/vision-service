<?php

use App\Http\Controllers\VisionController;
use Illuminate\Support\Facades\Route;

Route::middleware(['auth'])->group(function () {
    Route::post('batches/{batch}/vision/analyze', [VisionController::class, 'analyze'])
        ->middleware('can:update,batch')
        ->name('vision.analyze');
    Route::get('batches/{batch}/vision/logs', [VisionController::class, 'index'])->name('vision.logs');
});
