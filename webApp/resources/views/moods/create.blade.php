<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Moods</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <a href="{{ route('moods.index') }}">Back</a>
    <form action="{{ route('moods.store') }}" method="POST">
        @csrf
        <div class="m-3">
            <label for="mood">Mood</label>
            <input type="text" class="outline-1 @error('type') is-invalid @enderror" name="type" value="{{ old('type') }}">
            @error('type')
                <div class="invalid-feedback">
                    {{ $message }}
                </div>
            @enderror
        </div>
        <div class="m-3">
            <button type="submit" class="p-3 bg-red-500">Submit</button>
        </div>
    </form>
</body>
</html>
