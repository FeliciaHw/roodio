<?php

use Illuminate\Support\Facades\Route;

// Route::get('/', function () {
//     return view('welcome');
// });

Route::get('/', function () {
    return view('error/503');
});

Route::get('/login', function () {
    return view('auth/login');
});

Route::get('/sign-up', function () {
    return view('auth/register');
});

Route::get('/forget-password', function () {
    return view('auth/forgetPass');
});