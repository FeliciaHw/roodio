<?php

use App\Http\Controllers\AuthController;
use App\Http\Controllers\MoodController;
use App\Http\Controllers\SongController;
use App\Http\Controllers\ThreadController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('errors.503');
})->name('welcome');

Route::get('/songs/create', [SongController::class, 'create'])->name('songs.create');

// Auth Route
Route::get('/login', [AuthController::class, 'loginView'])->name('login');
Route::post('/login', [AuthController::class, 'login'])->name('auth.login');
Route::get('/sign-up', [AuthController::class, 'registerView'])->name('register');
Route::post('/sign-up', [AuthController::class, 'register'])->name('auth.register');
Route::get('/otp-authentication', [AuthController::class, 'registerValidationView'])->name('register.validation');
Route::post('/otp-authentication', [AuthController::class, 'registerValidation'])->name('auth.register.validation');
Route::get('/create-account', [AuthController::class, 'accountView'])->name('account');
Route::post('/create-account', [AuthController::class, 'account'])->name('auth.account');

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
Route::post('/threads/{thread}/reply', [ThreadController::class, 'reply'])->middleware('auth')->name('thread.reply');
// Route::post('/threads/{threadId}/reaction', [ThreadController::class, 'react'])->middleware('auth')->name('thread.react');

Route::get('/forget-password', function () {
    return view('auth/forgetPass');
});

Route::get('/develop', function () {
    return view('components.input');
});