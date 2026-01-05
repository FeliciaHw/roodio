@extends('layouts.signUp')


@section('title', 'ROODIO - Register')


@push('script')
    <script src="{{ asset('js/auth/register.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/logo-with-text.png') }}" alt="logo" class='w-20 drop-shadow-sm drop-shadow-primary-50'>
@endsection


@section('mainTitle', 'SIGN UP')


@section('description', "Let's join for amazing experience!")


{{-- @section('marginStyle', 'mt-16 pb-10 md:mt-18 md:pb-11 lg:mt-0 lg:pb-0') --}}
@section('customClass', 'lg:w-fit')

@section('content')
    <form action="{{ route('auth.register') }}" method="POST" id='identity'>
        @csrf
        <div class='lg:flex lg:flex-row lg:gap-1'>
            <div class='lg:w-[20rem]'>
                <x-input id='fullname' icon='name' label='Fullname' placeholder='Ex: John Doe' value="{{ old('fullname') }}"></x-input>
                <x-input type='email' id='email' icon='email' label='Email' placeholder='Ex: john.doe321@gmail.com' value="{{ old('email') }}"></x-input>
            </div>
            <div class='lg:w-[26rem]'>
                <div class='flex flex-row items-start justify-between'>
                    <x-input datepicker type='text' id='dob' icon='date' label='Date of Birth' placeholder='mm/dd/yyyy' value="{{ old('dob') }}"></x-input>
                    <x-inputSelect id='gender' icon='gender' label='Gender' class='gender-select' defaultOption='Your gender...'>
                        <x-slot:options>
                            <option value="1" {{ old('gender') === '1' ? 'selected' : '' }}>Male</option>
                            <option value="0" {{ old('gender') === '0' ? 'selected' : '' }}>Female</option>
                            <option value="null" {{ old('gender') === 'null' ? 'selected' : '' }}>Prefer not to say</option>
                        </x-slot:options>
                    </x-inputSelect>
                </div>
                <x-inputSelect id='country' icon='country' label='Country' class='country-select' defaultOption='Your country...'>
                    <x-slot:options>
                        @forelse($regions as $region)
                            <option value="{{ $region->id }}" {{ old('country') == $region->id ? 'selected' : '' }}>{{ $region->name }}</option>
                        @empty
                        @endforelse
                    </x-slot:options>
                </x-inputSelect>
            </div>
        </div>
        <div>
            <x-button behaviour='action' actionType='submit' form='identity' content='Sign Up' class='min-w-full'></x-button>
            <div class='flex flex-row justify-center gap-1'>
                <p class='text-micro text-center md:text-small'>Already have account?</p>
                <x-button behaviour='navigation' navType='text' :navLink="route('login')" content="Let's Login!"></x-button>
            </div>
        </div>
    </form>
@endsection
