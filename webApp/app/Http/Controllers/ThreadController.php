<?php
namespace App\Http\Controllers;

use App\Models\Reply;
use App\Models\Thread;
use function Symfony\Component\Clock\now;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class ThreadController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        $threads = Thread::orderByDesc('created_at')->paginate(5);
        return view('threads.index', compact('threads'));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        return view('threads.create');
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $request->validate([
            'title'   => 'required|max:100',
            'content' => 'required|max:255',
        ]);

        $threadData             = $request->all();
        $threadData['userId']   = Auth::id();
        $threadData['datePost'] = now();

        Thread::create($threadData);
        return redirect()->route('thread.index')->with('success', 'Thread added successfully!');
    }

    /**
     * Display the specified resource.
     */
    public function show(Thread $thread)
    {
        return view('threads.detail', compact('thread'));
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function reply(Request $request, Thread $thread)
    {
        $request->validate([
            'content' => "required|min:2|max:255",
        ]);

        $reply             = $request->all();
        $reply['userId']   = Auth::id();
        $reply['threadId'] = $thread->id;

        Reply::create($reply);
        return redirect()->route('thread.show', $thread)->with('succes', 'berhasil membuat reply');
    }
}
