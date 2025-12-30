<?php
namespace App\Http\Controllers;

use App\Mail\EmailOtp;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Mail;

class OtpController extends Controller
{
    public function send(string $email)
    {
        $otp = random_int(100000, 999999);

        Cache::put(
            "otp:{$email}",
            hash('sha256', $otp),
            now()->addMinutes(5)
        );

        Mail::to($email)->send(new EmailOtp($otp));
    }

    public function verify(string $email, string $inputOtp)
    {
        $key = "otp:{$email}";

        if (
            ! Cache::has($key) ||
            Cache::get($key) !== hash('sha256', $inputOtp)
        ) {
            return false;
        }

        Cache::forget($key);

        return true;
    }
}
