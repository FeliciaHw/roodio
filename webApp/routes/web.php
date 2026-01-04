<?php

use App\Http\Controllers\Admin\SongController;
use App\Http\Controllers\AuthController;
use App\Http\Controllers\MoodController;
use App\Http\Controllers\ThreadController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('errors.503');
})->name('welcome');

// Song Route
Route::prefix('admin/songs')->middleware(['auth', 'role:1'])->group(function () {
    Route::get('', [SongController::class, 'index'])->name('admin.songs.index');
    Route::get('/create', [SongController::class, 'create'])->name('admin.songs.create');
});

// Auth Route
Route::prefix('auth')->group(function () {
    Route::get('/login', [AuthController::class, 'loginView'])->name('login');
    Route::post('/login', [AuthController::class, 'login'])->name('auth.login');

    Route::get('/user-verification', [AuthController::class, 'userVerificationView'])->name('user.verification');
    Route::post('/user-verification', [AuthController::class, 'userVerification'])->name('auth.user.verification');
    Route::get('/forget-pasword', [AuthController::class, 'forgetPasswordView'])->name('forgetPassword');
    Route::post('/forget-password', [AuthController::class, 'forgetPassword'])->name('auth.forgetPassword');

    Route::get('/sign-up', [AuthController::class, 'registerView'])->name('register');
    Route::post('/sign-up', [AuthController::class, 'register'])->name('auth.register');

    Route::middleware('register.step:otp')->group(function () {
        Route::get('/otp-authentication', [AuthController::class, 'registerValidationView'])->name('register.validation');
        Route::post('/otp-authentication', [AuthController::class, 'registerValidation'])->name('auth.register.validation');
    });

    Route::middleware('register.step:account')->group(function () {
        Route::get('/create-account', [AuthController::class, 'accountView'])->name('account');
        Route::post('/create-account', [AuthController::class, 'account'])->name('auth.account');
    });
});

// Mood Route
Route::get('/moods', [MoodController::class, 'index'])->name('moods.index');
Route::get('/moods/create', [MoodController::class, 'create'])->name('moods.create');
Route::post('/moods', [MoodController::class, 'store'])->name('moods.store');
Route::get('/moods/{mood}/edit', [MoodController::class, 'edit'])->name('moods.edit');
Route::put('/moods/{mood}', [MoodController::class, 'update'])->name('moods.update');
Route::delete('/moods/{mood}', [MoodController::class, 'destroy'])->name('moods.destroy');

// Thread Route
Route::prefix('threads')->middleware('auth')->group(function () {
    Route::get('', [ThreadController::class, 'index'])->name('thread.index');
    Route::get('/create', [ThreadController::class, 'create'])->name('thread.create');
    Route::post('', [ThreadController::class, 'store'])->name('thread.store');
    Route::post('/{thread}/reply', [ThreadController::class, 'reply'])->name('thread.reply');
});
// Route::post('/{threadId}/reaction', [ThreadController::class, 'react'])->name('thread.react');

// Dev Route Preview
Route::get('/pageDevelop', function () {
    return view('components.navbarButton');
});
