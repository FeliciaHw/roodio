<?php

use App\Http\Controllers\AuthController;
use App\Http\Controllers\MoodController;
use App\Http\Controllers\SongController;
use App\Http\Controllers\ThreadController;

Route::get('/', function () {
    return view('welcome');
})->name('welcome');

Route::get('/songs/create', [SongController::class, 'create'])->name('songs.create');

// Auth Route
Route::get('/login', [AuthController::class, 'loginView'])->name('login');
Route::post('/login', [AuthController::class, 'login'])->name('auth.login');

// Mood Route
Route::get('/moods', [MoodController::class, 'index'])->name('moods.index');
Route::get('/moods/create', [MoodController::class, 'create'])->name('moods.create');
Route::post('/moods', [MoodController::class, 'store'])->name('moods.store');
Route::get('/moods/{mood}/edit', [MoodController::class, 'edit'])->name('moods.edit');
Route::put('/moods/{mood}', [MoodController::class, 'update'])->name('moods.update');
Route::delete('/moods/{mood}', [MoodController::class, 'destroy'])->name('moods.destroy');

// Thread Route
Route::get('/threads', [ThreadController::class, 'index'])->name('thread.index');
Route::get('/threads/create', [ThreadController::class, 'create'])->middleware('auth')->name('thread.create');
Route::post('/threads', [ThreadController::class, 'store'])->middleware('auth')->name('thread.store');

Route::get('/sign-up', function () {
    return view('auth/register');
});

Route::get('/forget-password', function () {
    return view('auth/forgetPass');
});
