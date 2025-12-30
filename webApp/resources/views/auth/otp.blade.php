@extends('layouts.signUp')


@section('title', 'ROODIO - OTP Authentication')


@push('script')
    <script src="{{ asset('js/auth/otp.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/icons/otp.svg') }}" alt="logo" class='w-24'>
@endsection


@section('mainTitle', 'OTP VERIFICATION')


@section('description')
    <span>Thank you for signing up with ROODIO! We have sent you the 6-digit code <b>to your email. Please enter the code to continue this signing up process.</b></span>
@endsection


@section('content')
    <form action="{{ route('auth.otp') }}" method="POST" id='otp-form'>
        @csrf
        <div class='flex flex-col mt-4 mb-8 otp-container'>
            <div class='flex flex-row justify-center gap-4 items-center'>
                @php ($amount = 6)
                @php ($idx = 1)
                @while ($idx <= $amount)
                    <input type="text" maxlength="1" inputmode="numeric" name="otp-{{ $idx }}" id="otp-{{ $idx }}" autocomplete="off" placeholder="*" class='not-placeholder-shown:bg-white bg-shadedOfGray-20 text-center text-paragraph outline-none border-2 font-bold rounded-md px-1.5 py-0.5 w-10 h-12 border-primary-30 placeholder:text-paragraph focus:border-secondary-happy-100 focus:bg-secondary-happy-20/50 ease-in-out duration-150 {{ $errors->has('otp') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
                    @php ($idx++)
                @endwhile
            </div>
            <div class="error-message">
                @error('otp')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <button type='submit' form='otp-form' class='text-smallBtn font-bold w-full font-secondaryAndButton bg-primary-10 text-primary-100 rounded-2xl py-1 mb-2 cursor-pointer hover:bg-primary-50 hover:text-white ease-in-out duration-150'>Verify Account</button>
        <p class='text-micro text-center md:text-small'>
            Don't get the code?
            <button class='font-bold text-secondary-sad-100 hover:text-primary-50'> Resend The Code.</button>
        </p>
    </form>
@endsection