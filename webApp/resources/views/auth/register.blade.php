@extends('layouts/signUp')


@section('title', 'ROODIO - Register')


@push('script')
    <script src="{{ asset('js/auth/register.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/logo-with-text.png') }}" alt="logo" class='w-24 drop-shadow-sm drop-shadow-primary-50'>
@endsection


@section('mainTitle', 'SIGN UP')


@section('description', "Let's join for amazing experience!")


@section('content')
    <form action="{{ route('auth.register') }}" method="POST" id='identity'>
        @csrf
        <x-input id='fullname' icon='name' label='Fullname' placeholder='Ex: John Doe'></x-input>
        <x-input type='email' id='email' icon='email' label='Email' placeholder='Ex: john.doe@gmail.com'></x-input>
        <div class='flex flex-col mb-6'>
            <label for="fullname" class='text-body-size flex flex-row mb-2'>
                <img src="{{ asset('assets/icons/name.svg') }}" alt="name" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Fullname<span class='text-danger'>*</span></p>
            </label>
            <input type="text" name="fullname" id="fullname" autocomplete="off" placeholder="Ex: John Doe" class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small {{ $errors->has('fullname') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}' value="{{ old('fullname') }}">
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
                @error('fullname')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div class='flex flex-col mb-6'>
            <label for="email" class='text-body-size flex flex-row mb-2'>
                <img src="{{ asset('assets/icons/email.svg') }}" alt="email" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Email<span class='text-danger'>*</span></p>
            </label>
            <input type="email" name="email" id="email" inputmode="email" autocomplete="off" placeholder="Ex: john.doe@gmail.com" class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small {{ $errors->has('email') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}' value="{{ old('email') }}">
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
                @error('email')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div class='flex flex-row items-start justify-between mb-6'>
            <x-input datepicker type='text' id='dob' icon='date' label='Date of Birth' placeholder='mm/dd/yyyy' class='grow'></x-input>
            {{-- <div class='flex flex-col'>
                <label for="dob" class='text-body-size flex flex-row mb-2'>
                    <img src="{{ asset('assets/icons/date.svg') }}" alt="email" class='w-6 mr-1'>
                    <p class='text-primary-85 text-body-size'>Date of Birth<span class='text-danger'>*</span></p>
                </label>
                <input datepicker id="default-datepicker" name='dob' type="text" autocomplete="off" placeholder="mm/dd/yyyy" class="not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 w-32 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small lg:w-42 {{ $errors->has('dob') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}" value="{{ old('dob') }}">
                <div class="text-error-moderate h-max pt-0.5 text-micro md:text-small">
                    @error('dob')
                        {{ $message }}
                    @enderror
                </div>
            </div> --}}
            <div class='flex flex-col grow-3'>
                <label for="gender" class='text-body-size flex flex-row mb-2'>
                    <img src="{{ asset('assets/icons/gender.svg') }}" alt="gender" class='w-6 mr-1'>
                    <p class='text-primary-85'>Gender<span class='text-danger'>*</span></p>
                </label>
                <select name="gender" id="gender" autocomplete="off" class="gender-select w-40 text-small outline-none border-b rounded-md px-1.5 py-1 border-shadedOfGray-50 text-shadedOfGray-60 italic focus:border-secondary-happy-100 focus:border-b-2 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 lg:w-44 {{ $errors->has('gender') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}">
                    <option value="" disabled hidden {{ old('gender') ? '' : 'selected' }}>
                        Gender
                    </option>

                    <option value="1" {{ old('gender') === '1' ? 'selected' : '' }}>
                        Male
                    </option>

                    <option value="0" {{ old('gender') === '0' ? 'selected' : '' }}>
                        Female
                    </option>

                    <option value="null" {{ old('gender') === 'null' ? 'selected' : '' }}>
                        Prefer not to say
                    </option>
                </select>
                <div class="text-error-moderate h-max pt-0.5 text-micro md:text-small">
                    @error('gender')
                        {{ $message }}
                    @enderror
                </div>
            </div>
        </div>
        <div class='flex flex-col mb-8'>
            <label for="country" class='text-body-size flex flex-row mb-2'>
                    <img src="{{ asset('assets/icons/country.svg') }}" alt="country" class='w-6 mr-1'>
                    <p class='text-primary-85'>Country<span class='text-danger'>*</span></p>
            </label>
            <select autocomplete="country" id="country" name="country" autocomplete="off" class="gender-select text-small outline-none border-b rounded-md px-1.5 py-1 border-shadedOfGray-50 text-shadedOfGray-60 italic focus:border-secondary-happy-100 focus:border-b-2 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 {{ $errors->has('country') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}">
                <option value="" disabled {{ old('country') ? '' : 'selected' }}>
                    Select your country...
                </option>
                @forelse($regions as $region)
                    <option value="{{ $region->id }}" {{ old('country') == $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                @empty
                @endforelse
            </select>
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
                @error('country')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div>
            <button type="submit" id='identityBtn' form='identity' class='text-smallBtn font-bold w-full font-secondaryAndButton bg-primary-10 text-primary-100 rounded-2xl py-1 mb-2 cursor-pointer hover:bg-primary-50 hover:text-white ease-in-out duration-150'>Sign Up</button>
            <p class='text-micro text-center md:text-small'>Already have account? <a href="/login" class='font-bold text-secondary-sad-100 hover:text-primary-50'>Let's Login!</a></p>
        </div>
    </form>
@endsection
