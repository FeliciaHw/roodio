<?php

use App\Http\Controllers\MoodController;
use App\Http\Controllers\SongController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('503error');
});

// Song Create
Route::get('/songs/create', [SongController::class, 'create'])->name('songs.create');

// Mood Route
Route::get('/moods', [MoodController::class, 'index'])->name('moods.index');
Route::get('/moods/create', [MoodController::class, 'create'])->name('moods.create');
Route::post('/moods', [MoodController::class, 'store'])->name('moods.store');
Route::get('/moods/{mood}/edit', [MoodController::class, 'edit'])->name('moods.edit');
Route::put('/moods/{mood}', [MoodController::class, 'update'])->name('moods.update');
Route::delete('/moods/{mood}', [MoodController::class, 'destroy'])->name('moods.destroy');
